import boto3
from botocore.config import Config
from botocore import UNSIGNED
from pathlib import Path
import pandas as pd
from collections import defaultdict
import aiohttp
import asyncio
from tqdm import tqdm
import tifffile
import aioboto3
from concurrent.futures import ThreadPoolExecutor
import h5py
from datetime import datetime
from asyncio import Semaphore
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default = 4)
    parser.add_argument("--samples_per_category", type=int, default = 250)
    parser.add_argument("--grade_level", type=int, default = 3)
    return parser.parse_args()

args = parse_args()

# Create a semaphore to limit concurrent API calls
# Ensembl allows 15 requests per second
API_SEMAPHORE = Semaphore(3)  # Conservative limit of 3 concurrent requests
RATE_LIMIT_DELAY = 0.2  # 200ms between requests to stay well under the limit

async def ensemble_to_gene(ensemble_id: str):
    """Convert Ensembl ID to gene symbol using BioMart REST API"""
    async with API_SEMAPHORE:
        try:
            # BioMart REST API URL
            url = "http://www.ensembl.org/biomart/martservice"
            
            # Construct XML query
            xml_query = f"""<?xml version="1.0" encoding="UTF-8"?>
                <!DOCTYPE Query>
                <Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" count="" datasetConfigVersion="0.6">
                    <Dataset name="hsapiens_gene_ensembl" interface="default">
                        <Filter name="ensembl_gene_id" value="{ensemble_id}"/>
                        <Attribute name="ensembl_gene_id"/>
                        <Attribute name="hgnc_symbol"/>
                    </Dataset>
                </Query>"""
            
            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params={'query': xml_query}) as response:
                    response.raise_for_status()
                    result = await response.text()
                    
                    # Parse result (tab-separated: ensembl_id, gene_symbol)
                    if result and '\t' in result:
                        gene_symbol = result.strip().split('\t')[1]
                        if gene_symbol:
                            await asyncio.sleep(RATE_LIMIT_DELAY)
                            return gene_symbol
            
            # Fallback to original ID if no result
            print(f"No symbol found for {ensemble_id}, using ID as fallback")
            return ensemble_id
            
        except Exception as e:
            print(f"Error converting {ensemble_id}: {e}")
            return ensemble_id


class OpenCellDataLoader:
    def __init__(self, base_dir="opencell_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "images").mkdir(exist_ok=True)
        
        # S3 client for microscopy images
        self.session = aioboto3.Session()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.bucket = 'czb-opencell'
        
        # Thread pool for sync operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
    async def download_images(self, ensemble_id: str):
        gene_name = await ensemble_to_gene(ensemble_id)
        if not gene_name:
            print(f"Could not get gene name for {ensemble_id}")
            return
            
        prefix = f"microscopy/raw/{gene_name}_{ensemble_id}/"
        output_path = self.base_dir / "images" / ensemble_id
        if not output_path.exists():
            output_path.mkdir(parents=True)
            
        try:
            async with self.session.client(
                's3',
                config=Config(signature_version=UNSIGNED)
            ) as s3:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=prefix
                )
                
                if 'Contents' in response:
                    download_tasks = []
                    for obj in response['Contents']:
                        key = obj['Key']
                        if 'proj' in key:
                            protein_output_path = output_path / Path(key).name
                            if not protein_output_path.exists():
                                download_tasks.append(
                                    self._download_file(s3, key, str(protein_output_path))
                                )
                    
                    if download_tasks:
                        await asyncio.gather(*download_tasks)
                else:
                    print(f"No images found for {gene_name} ({ensemble_id})")
                    
        except Exception as e:
            print(f"Error downloading images for {ensemble_id}: {e}")

    async def _download_file(self, s3_client, key: str, output_path: str):
        try:
            await s3_client.download_file(
                Bucket=self.bucket,
                Key=key, 
                Filename=output_path
            )
        except Exception as e:
            print(f"Error downloading {key}: {e}")


loader = OpenCellDataLoader()
    
    
async def prepare_localization_data(
    metadata_path: Path = Path("opencell_data/metadata/opencell-localization-annotations.csv"),
    num_categories: int = 4,
    samples_per_category: int = 30,
    grade_level: int = 3,
    output_file: str = "protein_localizations.h5",
    multi_class: bool = False
):
    """Gather data from OpenCell S3 bucket and organize images with their localizations"""
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata[f'annotations_grade_{grade_level}'].notna()]
    protein_localizations = defaultdict(list)
    
    # Process annotations
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing annotations"):
        protein_location = row[f'annotations_grade_{grade_level}']
        if ';' in protein_location:
            if multi_class:
                protein_locations = protein_location.split(';')
                for protein_location in protein_locations:
                    protein_localizations[protein_location.strip()].append(row['ensg_id'])

        else:
            protein_localizations[protein_location.strip()].append(row['ensg_id'])
    
    # Sort and filter localizations
    protein_localizations = sorted(protein_localizations.items(), key=lambda x: len(x[1]), reverse=True)
    protein_localizations = [(category, sample_ids[:samples_per_category]) if len(sample_ids) >= samples_per_category 
                           else (category, sample_ids) for category, sample_ids in protein_localizations]
    protein_localizations = protein_localizations[:num_categories]

    print(f'Selected locations: {[i[0] for i in protein_localizations]}')

    # Download images
    tasks = []
    for _, sample_ids in tqdm(protein_localizations, desc="Creating download tasks"):
        if isinstance(sample_ids, tuple):
            sample_ids = sample_ids[1] 
        for sample_id in sample_ids:
            tasks.append(loader.download_images(ensemble_id=sample_id))
    
    await asyncio.gather(*tasks)

    # Create HDF5 file to store the data
    total_files_processed = 0
    with h5py.File(output_file, 'w') as f:
        # Create groups for each localization
        for category, sample_ids in tqdm(protein_localizations, desc="Processing images"):
            if isinstance(sample_ids, tuple):
                sample_ids = sample_ids[1]
                
            # Create a group for this localization category
            category_group = f.create_group(category)
            
            # Store images for each sample
            for sample_id in sample_ids:
                image_path = loader.base_dir / "images" / sample_id
                if image_path.exists():
                    # Get all tif files in the directory
                    tif_files = sorted(list(image_path.glob("*.tif")))
                    
                    # Create a group for this sample
                    sample_group = category_group.create_group(sample_id)
                    
                    # Store metadata
                    sample_group.attrs['num_images'] = len(tif_files)
                    sample_group.attrs['date_processed'] = datetime.now().isoformat()
                    
                    # Store each image with a unique ID
                    for idx, tif_file in enumerate(tif_files, 1):
                        try:
                            # Read image data
                            image_data = tifffile.imread(str(tif_file))
                            
                            # Create unique ID: sample_id_imagenum
                            image_id = f"{sample_id}_{idx}"
                            
                            # Store image data
                            sample_group.create_dataset(
                                f"image_{idx}", 
                                data=image_data,
                                compression="gzip",
                                compression_opts=9
                            )
                            
                            # Store image metadata
                            sample_group[f"image_{idx}"].attrs['image_id'] = image_id
                            sample_group[f"image_{idx}"].attrs['filename'] = tif_file.name
                            total_files_processed += 1
                        except Exception as e:
                            print(f"Error processing {tif_file}: {e}")
        print(f'Total files processed: {total_files_processed}')
        
    return protein_localizations


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(prepare_localization_data(
        num_categories = arguments.categories,
        samples_per_category = arguments.samples_per_category,
        grade_level = arguments.grade_level
    ))