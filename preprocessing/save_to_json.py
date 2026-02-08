import pandas as pd
import json


def preprocess_to_json(
    data_path: str, meta_data_cols: list[str], text_cols: list[str], output_path: str
) -> None:
    """
    Preprocesses a CSV dataset and exports it as a JSON file.

    For each row in the CSV, creates a dictionary with two keys:
    - 'text_data': Contains columns specified in text_cols
    - 'meta_data': Contains columns specified in meta_data_cols

    Args:
        data_path: Path to the input CSV file
        meta_data_cols: List of column names to include in meta_data
        text_cols: List of column names to include in text_data
        output_path: Path where the output JSON file will be saved

    Returns:
        None. Saves the processed data to output_path as JSON.

    Example:
        >>> preprocess_to_json(
        ...     data_path='data.csv',
        ...     meta_data_cols=['id', 'date'],
        ...     text_cols=['title', 'description'],
        ...     output_path='output.json'
        ... )
    """
    # 1. Load the CSV dataset
    df = pd.read_csv(data_path)

    # Validate that all specified columns exist in the dataframe
    all_cols = set(meta_data_cols + text_cols)
    missing_cols = all_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"The following columns are not in the dataset: {missing_cols}"
        )

    # Process each row and create the desired structure
    result = []

    for _, row in df.iterrows():
        # 2. Create text_data JSON from text_cols
        text_data = {col: row[col] for col in text_cols}

        # 3. Create meta_data JSON from meta_data_cols
        meta_data = {col: row[col] for col in meta_data_cols}

        # 4. Combine into final structure
        row_dict = {"text_data": text_data, "meta_data": meta_data}

        result.append(row_dict)

    # Export the final data as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(result)} rows and saved to {output_path}")


if __name__ == "__main__":
    preprocess_to_json(
        data_path="/Users/mac/Desktop/ctg-studies-2.csv",
        meta_data_cols=[
            "NCT Number",
            "Study Status",
            "Study Results",
            "Conditions",
            "Study Type",
            "Sex",
            "Age",
        ],
        text_cols=["Brief Summary", "Study Status", "Conditions"],
        output_path="/Users/mac/Desktop/Projects/clinical_trials/data/clean_data.json",
    )
