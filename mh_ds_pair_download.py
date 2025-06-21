import pandas as pd
from tqdm import tqdm
import requests
import os

def prepareDataset(ds="cifar10"):
    ##  Read the CSV file
    df = pd.read_csv(f'{ds}_experiment_results.csv')
    ##  Transform the 'image_link' column to only keep the part after 'IMG_TRANSFORMED'
    df["image_id"] = df["image_link"].map(lambda x: x[x.find('IMG'):])
    df["ground_truth"] = df["ground_truth"].map(lambda x: 0 if x != "car" else 1)
    df["human_label"] = df["human_label"].map(lambda x: 0 if x != "car" else 1)
    df_original = df[df["transformation"] == "original"]
    df_transformed = df[df["transformation"] != "original"]
    df_original["image_id"] = df_original["image_id"].map(lambda x: x[x.find('/')+1:])
    df_transformed["image_id"] = df_transformed["image_id"].map(lambda x: x[x.find('/')+1:])
    ##  Remove duplicate rows based on 'image_id' in df_original
    print("Before removing duplicates:", df_original.shape)
    df_original = df_original.drop_duplicates(subset=["image_id"])
    print("After removing duplicates:", df_original.shape)
    df_original["transformed_link"] = ""
    df_original["transformed_label"] = ""
    df_original["invalid"] = 0

    ##  Putting the transformed links and labels into the original DataFrame for each image_id
    for index, row in df_original.iterrows():
        image_id = row["image_id"]
        df_filtered = df_transformed[df_transformed["image_id"] == image_id]
        if not df_filtered.empty:
            df_original.loc[index, "transformed_link"] = df_filtered["image_link"].values[0]
            ##  If the human_label is different in different rows of df_filtered, we take the opposite of the ground_truth
            # df_original.loc[index, "transformed_label"] = df_filtered["human_label"].values[0]
            if df_filtered["human_label"].nunique() > 1:
                df_original.loc[index, "transformed_label"] = 1 - row["ground_truth"]
                df_original.loc[index, "invalid"] = 1
            else:
                df_original.loc[index, "transformed_label"] = df_filtered["human_label"].values[0]
                df_original.loc[index, "invalid"] = 0
            

    ##  Remove rows where transformed_link is empty
    df_original = df_original[df_original["transformed_link"] != ""]

    n_valid = df_original[df_original["invalid"] == 0].shape[0]
    n_invalid = df_original[df_original["invalid"] == 1].shape[0]
    print(f"Number of valid pairs: {n_valid} / {df_original.shape[0]} ({n_valid / df_original.shape[0] * 100:.2f}%)")
    print(f"Number of invalid pairs: {n_invalid} / {df_original.shape[0]} ({n_invalid / df_original.shape[0] * 100:.2f}%)")
    columns = ["image_id", "image_link", "transformed_link", "ground_truth", "transformed_label", "invalid"]
    df_original = df_original[columns]
    df_original.to_csv(f"{ds}_prepared_dataset.csv", index=False)
    print(f"Prepared dataset saved to {ds}_prepared_dataset.csv")

def downloadDataset(csv_file, output_folder='Pair_Hu_Cifar10'):
    if os.path.exists(output_folder):
        print(f"Output folder '{output_folder}' already exists. Please remove it or choose a different name.")
        return
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "transformed"), exist_ok=True)
    df = pd.read_csv(csv_file)
    errors = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_link = row["image_link"]
        transformed_link = row["transformed_link"]
        image_id = row["image_id"]
        
        # Download original image
        if pd.notna(image_link) and image_link:
            try:
                response = requests.get(image_link)
                if response.status_code == 200:
                    with open(os.path.join(output_folder, "original", image_id.replace('/', '_')), 'wb') as f:
                        f.write(response.content)
            except Exception as e:
                print(f"Error downloading {image_link}: {e}")
                errors.append((image_id, image_link, str(e)))

        # Download transformed image
        if pd.notna(transformed_link) and transformed_link:
            try:
                response = requests.get(transformed_link)
                if response.status_code == 200:
                    with open(os.path.join(output_folder, "transformed", image_id.replace('/', '_')), 'wb') as f:
                        f.write(response.content)
            except Exception as e:
                print(f"Error downloading {transformed_link}: {e}")
                errors.append((image_id, transformed_link, str(e)))

    if errors:
        print("Errors occurred while downloading images:")
        for error in errors:
            print(f"Image ID: {error[0]}, Link: {error[1]}, Error: {error[2]}")
if __name__ == "__main__":
    ds = "imagenet"
    prepareDataset(ds)
    downloadDataset(f"{ds}_prepared_dataset.csv")