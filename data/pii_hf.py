import re
import requests
from bs4 import BeautifulSoup
import csv


def list_pii_datasets():
    """Returns first page of results for PII-related datasets on huggingface.co sorted by most rows"""
    url = "https://huggingface.co/datasets?sort=most_rows&search=pii"

    response = requests.get(url)
    soup = BeautifulSoup(response.content.decode("utf8"), features="html.parser")

    for dataset in soup.find_all("article"):
        parsed_text = [
            line.strip()
            for line in re.sub(
                " +",
                " ",
                dataset.text.replace("\n", " ").replace("\t", " ").replace("•", "\n"),
            )
            .strip()
            .split("\n")
        ]

        # Extract dataset name and details
        dataset_name_str, *details = parsed_text
        dataset_name = dataset.find("a").attrs["href"][1:]

        # Remove 'datasets/' prefix if present
        if dataset_name.startswith('datasets/'):
            dataset_name = dataset_name[len('datasets/'):]

        num_rows = details[1] if details else 'NA'

        # Extract additional information from the dataset's specific page
        dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
        dataset_page = requests.get(dataset_url)
        dataset_soup = BeautifulSoup(dataset_page.content.decode("utf8"), features="html.parser")

        # Initialize values with default NA or specified defaults
        license_type, data_size, version, num_classes, data_format, language, data_type = "NA", "NA", "1.0", "NA", "csv", "English", "NA"

        # Extract the text content for parsing size and number of rows
        full_page_text = dataset_soup.get_text().lower()

        # Regex to extract the size in GB or MB and rows information
        size_match = re.search(r"(\d+(?:\.\d+)?)\s*(gb|mb)size", full_page_text)
        num_rows_match = re.search(r"number of rows:\s*([\d,]+)", full_page_text)

        if size_match:
            data_size = f"{size_match.group(1)} {size_match.group(2).upper()}"
        if num_rows_match:
            num_rows = num_rows_match.group(1).replace(",", "")  # Remove commas for numeric consistency

        # Check for other details like license, version, etc.
        for section in dataset_soup.find_all("section"):
            section_text = section.text.lower()
            if "license" in section_text:
                license_type = section.find("span", class_="badge").text if section.find("span",
                                                                                         class_="badge") else "NA"
            if "version" in section_text:
                version = section_text.split("version:")[-1].strip().split()[0]
            if "classes" in section_text:
                num_classes = section_text.split("classes:")[-1].strip().split()[0]
            if "data format" in section_text:
                data_format = section_text.split("data format:")[-1].strip().split()[0]
            if "language" in section_text:
                language = section_text.split("language:")[-1].strip().split()[0]
            if "data type" in section_text:
                data_type = section_text.split("data type:")[-1].strip().split()[0]

        # Set default values for missing fields
        language = language if language else "English"
        data_format = data_format if data_format else "csv"
        version = version if version else "1.0"

        yield {
            "dataset_name": dataset_name,
            "num_rows": num_rows,
            "license": license_type,
            "url": dataset_url,
            "data_size": data_size,
            "version": version,
            "num_classes": num_classes,
            "data_format": data_format,
            "language": language,
            "data_type": data_type,
        }


# Fetch and save the top 10 PII-related datasets sorted by most rows in CSV format
with open('pii_datasets.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=[
        "dataset_name", "num_rows", "license", "url", "data_size", "version", "num_classes",
        "data_format", "language", "data_type"
    ])
    writer.writeheader()

    for i, dataset in enumerate(list_pii_datasets()):
        if i == 10:
            break
        writer.writerow(dataset)

print("CSV file 'pii_datasets.csv' created successfully.")
