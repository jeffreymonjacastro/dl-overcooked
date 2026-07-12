import pathlib
import zipfile

def main():
    data_dir = pathlib.Path("data")
    zip_path = pathlib.Path("kaggle_staging/data.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Zipping npz files...")
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for npz_path in data_dir.rglob("*.npz"):
            # Use relative path inside the zip file
            rel_path = npz_path.relative_to(data_dir.parent)
            zipf.write(npz_path, rel_path)
            count += 1
            if count % 50 == 0:
                print(f"  Zipped {count} files...")
    
    print(f"Done! Zipped {count} files to {zip_path}")

if __name__ == "__main__":
    main()
