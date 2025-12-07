# pip install docling
# pip install docling[asr,easyocr]
# pip install easyocr
# pip install hf_xet # for better performance than http
# pip install pillow # used for image export

# Prefetch the models
# explicitly prefetch them for offline use (e.g. in air-gapped environments)
# cmd:
# $ docling-tools models download
# Downloading layout model...
# Downloading tableformer model...
# Downloading picture classifier model...
# Downloading code formula model...
# Downloading easyocr models...
# Models downloaded into $HOME/.cache/docling/models.
#
# or
#
# in python:
# docling.utils.model_downloader.download_models()
#
# Also, you can use download-hf-repo parameter to download arbitrary models from HuggingFace by specifying repo id:
# $ docling-tools models download-hf-repo ds4sd/SmolDocling-256M-preview
import argparse
import asyncio
import logging
import time
from collections.abc import Iterable
from pathlib import Path
import shutil


from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.accelerator_options import AcceleratorOptions

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableFormerMode,
)
from docling.datamodel.pipeline_options import granite_picture_description
from docling.document_converter import DocumentConverter, PdfFormatOption
import mlflow
from utils.params_helper import load_params
from utils.device_helper import get_device
from utils.mlflow_helper import mlflow_connect
from utils.mlflow_helper import get_mlflow_experiment_name
from datetime import datetime

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0


def export_documents(
    conv_results: Iterable[ConversionResult], output_dir: Path, source_handling: str
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0
    conv_count = 0

    json_doc_counter = 0
    html_doc_counter = 0
    doctags_doc_counter = 0
    markdown_doc_counter = 0

    for conv_res in conv_results:
        conv_count += 1
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            mlflow.log_metric("success_count", success_count, step=conv_count)
            doc_filename = conv_res.input.file.stem
            filesize = conv_res.input.filesize if conv_res.input.filesize else 0
            mlflow.log_metric("file_size_bytes", float(filesize), step=conv_count)

            # Save page
            SAVE_PAGE_IMAGES = False
            mlflow.log_metric(
                "save_page_images", int(SAVE_PAGE_IMAGES), step=conv_count
            )
            if SAVE_PAGE_IMAGES:
                images_counter = 0
                for page_no, page in conv_res.document.pages.items():
                    page_no = page.page_no
                    page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
                    with page_image_filename.open("wb") as fp:
                        if page.image and page.image.pil_image:
                            page.image.pil_image.save(fp, format="PNG")
                            images_counter += 1
                mlflow.log_metric("images_count", images_counter, step=conv_count)

            # Save images of figures and tables
            SAVE_IMAGES_OF_FIGURES = False
            SAVE_IMAGES_OF_TABLES = False
            mlflow.log_metric(
                "save_images_of_figures", int(SAVE_IMAGES_OF_FIGURES), step=conv_count
            )
            mlflow.log_metric(
                "save_images_of_tables", int(SAVE_IMAGES_OF_TABLES), step=conv_count
            )
            if SAVE_IMAGES_OF_FIGURES or SAVE_IMAGES_OF_TABLES:
                table_counter = 0
                picture_counter = 0
                for element, _level in conv_res.document.iterate_items():
                    if SAVE_IMAGES_OF_TABLES and isinstance(element, TableItem):
                        table_counter += 1
                        element_image_filename = (
                            output_dir / f"{doc_filename}-table-{table_counter}.png"
                        )
                        with element_image_filename.open("wb") as fp:
                            img = element.get_image(conv_res.document)
                            if img:
                                img.save(fp, "PNG")

                    if SAVE_IMAGES_OF_FIGURES and isinstance(element, PictureItem):
                        picture_counter += 1
                        element_image_filename = (
                            output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                        )
                        with element_image_filename.open("wb") as fp:
                            img = element.get_image(conv_res.document)
                            if img:
                                img.save(fp, "PNG")
                mlflow.log_metric("tables_images_count", table_counter, step=conv_count)
                mlflow.log_metric(
                    "figures_images_count", picture_counter, step=conv_count
                )

            # Recommended modern Docling exports. These helpers mirror the
            # lower-level "export_to_*" methods used below, but handle
            # common details like image handling.
            conv_res.document.save_as_json(
                output_dir / f"{doc_filename}.json",
                image_mode=ImageRefMode.REFERENCED,
            )
            json_doc_counter += 1
            mlflow.log_metric("json_docs_count", json_doc_counter, step=conv_count)
            conv_res.document.save_as_html(
                output_dir / f"{doc_filename}.html",
                image_mode=ImageRefMode.EMBEDDED,  # ImageRefMode.REFERENCED
            )
            html_doc_counter += 1
            mlflow.log_metric("html_docs_count", html_doc_counter, step=conv_count)
            conv_res.document.save_as_doctags(
                output_dir / f"{doc_filename}.doctags.txt"
            )
            doctags_doc_counter += 1
            mlflow.log_metric(
                "doctags_docs_count", doctags_doc_counter, step=conv_count
            )
            conv_res.document.save_as_markdown(
                output_dir / f"{doc_filename}.md",
                image_mode=ImageRefMode.REFERENCED,
            )
            markdown_doc_counter += 1
            mlflow.log_metric(
                "markdown_docs_count", markdown_doc_counter, step=conv_count
            )

            # move or copy source file
            if source_handling == "copy":
                print("copy the source file to the target directory")
                shutil.copyfile(
                    conv_res.input.file, output_dir / conv_res.input.file.name
                )

            elif source_handling == "move":
                print("move the source file to the target directory")
                shutil.move(conv_res.input.file, output_dir / conv_res.input.file.name)

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
            mlflow.log_metric(
                "partial_success_count", partial_success_count, step=conv_count
            )
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1
            mlflow.log_metric("failure_count", failure_count, step=conv_count)

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count


async def process_documents(
    pdf_path: Path, out_dir: Path, batch_size: int, source_handling: str
):
    logging.basicConfig(level=logging.INFO)

    print(pdf_path)
    input_doc_paths = None

    if pdf_path.is_dir():
        input_doc_paths = [f for f in pdf_path.iterdir() if f.is_file()]
    else:
        input_doc_paths = [pdf_path]

    # pipeline_options = PipelineOptions()
    # artifacts_path = "/local/path/to/models"
    # pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path, do_table_structure=True)

    pipeline_options = PdfPipelineOptions()
    # Explicitly set the accelerator
    accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=get_device(),  # AcceleratorDevice.CPU #AUTO
    )
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.ocr_batch_size = 8
    pipeline_options.layout_batch_size = 32
    pipeline_options.table_batch_size = 16

    pipeline_options.do_code_enrichment = True

    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.do_picture_classification = True

    pipeline_options.picture_description_options = (
        granite_picture_description  # <-- the model choice
    )
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options.prompt = (
        "Describe the image in three sentences. Be consise and accurate."
    )

    pipeline_options.do_formula_enrichment = True

    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = (
        False  # uses text cells predicted from table structure model
    )
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
    )  # use more accurate TableFormer model

    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = (
        EasyOcrOptions()
    )  # TesseractOcrOptions()  # Use Tesseract

    pipeline_options.enable_remote_services = (
        True  # PictureDescriptionApiOptions: Using vision models via API calls.
    )

    # Log all pipeline options to MLflow
    mlflow.log_param("device", get_device())
    mlflow.log_param("num_threads", accelerator_options.num_threads)
    mlflow.log_param("ocr_batch_size", pipeline_options.ocr_batch_size)
    mlflow.log_param("layout_batch_size", pipeline_options.layout_batch_size)
    mlflow.log_param("table_batch_size", pipeline_options.table_batch_size)
    mlflow.log_param("do_code_enrichment", pipeline_options.do_code_enrichment)
    mlflow.log_param("generate_page_images", pipeline_options.generate_page_images)
    mlflow.log_param(
        "generate_picture_images", pipeline_options.generate_picture_images
    )
    mlflow.log_param("images_scale", pipeline_options.images_scale)
    mlflow.log_param(
        "do_picture_classification", pipeline_options.do_picture_classification
    )
    mlflow.log_param("do_picture_description", pipeline_options.do_picture_description)
    mlflow.log_param(
        "picture_description_prompt",
        pipeline_options.picture_description_options.prompt,
    )
    mlflow.log_param("do_formula_enrichment", pipeline_options.do_formula_enrichment)
    mlflow.log_param("do_table_structure", pipeline_options.do_table_structure)
    mlflow.log_param(
        "table_structure_mode", str(pipeline_options.table_structure_options.mode)
    )
    mlflow.log_param(
        "do_cell_matching", pipeline_options.table_structure_options.do_cell_matching
    )
    mlflow.log_param("do_ocr", pipeline_options.do_ocr)
    mlflow.log_param("ocr_type", type(pipeline_options.ocr_options).__name__)
    mlflow.log_param("enable_remote_services", pipeline_options.enable_remote_services)

    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ],  # whitelist formats, non-matching files are ignored.
        # pipeline_options=pipeline_options,
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
            )
        },
    )

    start_time = time.time()

    # batch handling
    while len(input_doc_paths) > 0:
        batch: list[Path] = []
        while len(input_doc_paths) > 0 and len(batch) < batch_size:
            batch.append(input_doc_paths.pop(0))

        print(batch)
        mlflow.log_param("batch_count", len(batch))
        mlflow.log_param("input_doc_paths_count", len(input_doc_paths))

        # Convert all inputs. Set `raises_on_error=False` to keep processing other
        # files even if one fails; errors are summarized after the run.
        conv_results = doc_converter.convert_all(
            source=batch,
            raises_on_error=False,  # to let conversion run through all and examine results at the end
        )

        _success_count, _partial_success_count, failure_count = export_documents(
            conv_results=conv_results,
            output_dir=out_dir,
            source_handling=source_handling,
        )

        mlflow.log_param("success_count", _success_count)
        mlflow.log_param("partial_success_count", _partial_success_count)
        mlflow.log_param("failure_count", failure_count)

        # if failure_count > 0:
        #     raise RuntimeError(
        #         f"The example failed converting {failure_count} on {len(batch)}."
        #     )

        # if _partial_success_count > 0:
        #     raise RuntimeError(
        #         f"The example partially failed converting {_partial_success_count} on {len(batch)}."
        #     )

        # # move or copy source file
        # if(source_handling == "copy"):
        #     print("copy the source file to the target directory")
        #     for b in batch:
        #         filepath = Path(b)
        #         shutil.copyfile(filepath, out_dir / filepath.name)

        # elif(source_handling == "move"):
        #     print("mpve the source file to the target directory")
        #     #shutil.move(path, out_dir_per_file / path.name)

        end_time = time.time() - start_time
        _log.info(
            f"Document conversion complete in {end_time:.2f} seconds. it successfully completed {_success_count} out of {len(input_doc_paths)}"
        )


async def main(pdf_path: Path, out_dir: Path) -> None:
    extract_params = load_params("extract")

    batch_size: int = int(extract_params.get("batch_size", 1))
    mlflow.log_param("batch_size", batch_size)
    source_handling: str = str(extract_params.get("source_handling", ""))
    mlflow.log_param("source_handling", source_handling)

    await process_documents(pdf_path, out_dir, batch_size, source_handling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    mlflow_connect()
    experiment_name = get_mlflow_experiment_name()
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"extract_documents_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        asyncio.run(main(args.input.resolve(), args.output.resolve()))
