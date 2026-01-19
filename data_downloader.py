import gdown
import zipfile
import os

file_id = '1bpYfksqKwGzaxm4htFtQr0iCaWfQBRKZ'
download_url = f'https://drive.google.com/uc?id={file_id}'
output_zip = 'downloaded_file.zip'
gdown.download(download_url, output_zip, quiet=False)


with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall()  
    print("File unzipped successfully.")

os.remove(output_zip)
print(f"Zip file '{output_zip}' has been removed.")