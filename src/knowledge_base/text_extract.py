import string
import pymupdf
import docx
import pandas as pd
from typing import BinaryIO, Optional


INCLUDE_CHARS = set(string.ascii_letters + string.digits + string.punctuation)
TEXT_FILES = set(('.txt', '.md', '.htm', '.html'))
PYMUPDF_FILES = set(('.pdf', '.xps', '.epub', '.mobi', '.fb2', '.cbz', '.svg', '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.pnm', '.pgm', '.pbm', '.ppm', '.pam', '.jpx', '.jp2', '.psd'))
DOCX_FILES = set(('.docx',))
EXCEL_FILES = set(('.xlsx', '.xls', '.xlsm', '.xlsb'))


def extract_text(file: BinaryIO, filename: Optional[str] = None) -> str:
    """
    Extract text from a file.

    Args:
        file: A file-like object containing the file data
        filename: Optional filename to determine file type

    Returns:
        Extracted text from the file
    """
    # Determine file type from filename or file object
    if filename:
        file_ext = '.' + filename.lower().split('.')[-1]
    else:
        file_ext = '.pdf'  # Default to PDF if no filename provided

    try:
        if file_ext in PYMUPDF_FILES:
            return pymupdf_to_string(file)
        elif file_ext in DOCX_FILES:
            return docx_to_string(file)
        elif file_ext in TEXT_FILES:
            return textfile_to_string(file)
        elif file_ext in EXCEL_FILES:
            return excel_to_string(file)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    except Exception as e:
        raise Exception(f"Error extracting text from {file_ext} file: {str(e)}")


def textfile_to_string(file):
    """
        Loads a text file into a string.

        Returns: A string of the document's text contents.
    """
    return file.read().decode()


def docx_to_string(file):
    """
        Converts a .docx file to a string.

        Returns: A string of the document's text contents.
    """
    # TODO: This can be greatly improved -- just trying to get some formatting.
    # Likely there will be a lot of cases where this doesn't render well.
    # This really should be recursively traversing the document tree.
    doc = docx.Document(file)

    parts = []
    newline = False
    for elem in doc.element.iter():
        if type(elem) is docx.oxml.text.run.CT_Text:
            if newline:
                parts.append("\n")
            else:
                newline = True
            parts.append(elem.text)
        elif type(elem) is docx.oxml.table.CT_Tc:
            parts.append("\t")
            newline = False
        elif type(elem) is docx.oxml.table.CT_Row:
            parts.append("\n")
            newline = True
        elif type(elem) is docx.oxml.numbering.CT_NumPr:
            parts.append("\n* ")
            newline = False

    return "".join(parts)


def pymupdf_to_string(file):
    """
        Converts fitz supported file formats to a string.

        Returns: A string of the document's text contents.
    """
    try:
        stream = bytearray(file.read())
        doc = pymupdf.open(stream=stream)
        lines = []
        for page in doc:
            for block in page.get_text("blocks", sort=True):
                line = block[4].replace('\n', ' ')
                lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        raise Exception(f"Error processing PDF file: {str(e)}")


def excel_to_string(file: BinaryIO) -> str:
    """
    Converts an Excel file to a string representation.
    Handles multiple sheets and preserves table structure.

    Args:
        file: A file-like object containing the Excel data

    Returns:
        A string representation of the Excel file's contents
    """
    try:
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(file)
        sheets = excel_file.sheet_names

        text_parts = []

        for sheet_name in sheets:
            # Read each sheet into a DataFrame
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Add sheet name as header
            text_parts.append(f"\nSheet: {sheet_name}\n")

            # Convert DataFrame to string with proper formatting
            # Replace NaN with empty string
            df = df.fillna('')

            # Get column names
            columns = df.columns.tolist()
            text_parts.append(" | ".join(str(col) for col in columns))
            text_parts.append("-" * 80)  # Separator line

            # Add each row
            for _, row in df.iterrows():
                row_text = " | ".join(str(cell) for cell in row)
                text_parts.append(row_text)

            text_parts.append("\n")  # Add space between sheets

        return "\n".join(text_parts)
    except Exception as e:
        raise Exception(f"Error extracting text from Excel file: {str(e)}")