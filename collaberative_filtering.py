from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as npy
from scipy.stats import linregress
from sklearn.model_selection import GroupShuffleSplit
import time
from pathlib import Path
import traceback
import os
import gc
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knn_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger
logger = logging.getLogger(__name__)


logger.info("KNN Processing script started - logging configured")

def process_dataset(dataset_name, data, k_values, base_output_dir="knn/results"):
    """
    Process a single dataset with multiple k values
    Uses chunking for memory-efficient average rating calculation
    """
    logger.info(f"{'='*60}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*60}")
    
    dataset_start_time = time.time()
    
    try:
        # Data preprocessing
        logger.info(f"Preprocessing {dataset_name}...")
        data = data[data['user_id'].notnull()]
        
        # Map item IDs to continuous range
        unique_items = data['item_id'].unique()
        item_id_map = {old: new for new, old in enumerate(unique_items)}
        data['item_id'] = data['item_id'].map(item_id_map)
        n_items = len(unique_items)
        
        # Split data
        groups = data['user_id']
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(gss.split(data, groups=groups))
        
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Map user IDs for training data
        unique_users = train_data['user_id'].unique()
        user_id_map = {old: new for new, old in enumerate(unique_users)}
        train_data['user_id'] = train_data['user_id'].map(user_id_map)
        n_training_users = len(unique_users)
        
        # Create training matrix
        training_X = sparse.csr_matrix((train_data["rating"],
                                      (train_data["user_id"],
                                       train_data["item_id"])),
                                      shape=(n_training_users, n_items))
        
        # Filter test data for users with at least 2 ratings
        user_counts = test_data['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 2].index
        filtered_data = test_data[test_data['user_id'].isin(valid_users)]
        
        # Map user IDs for test data
        unique_test_users = filtered_data['user_id'].unique()
        test_user_id_map = {old: new for new, old in enumerate(unique_test_users)}
        filtered_data['user_id'] = filtered_data['user_id'].map(test_user_id_map)
        
        # Split test data into seen and unseen
        test_seen, test_unseen = train_test_split(
            filtered_data,
            test_size=0.25,
            stratify=filtered_data['user_id'],
            random_state=42
        )
        
        n_testing_users = test_seen['user_id'].nunique()
        
        # Create testing matrix
        testing_X = sparse.csr_matrix((test_seen["rating"],
                                     (test_seen["user_id"],
                                      test_seen["item_id"])),
                                     shape=(n_testing_users, n_items))
        
        logger.info(f"Dataset {dataset_name} preprocessed successfully")
        logger.info(f"Training users: {n_training_users}, Testing users: {n_testing_users}, Items: {n_items}")
        
        # Process each k value
        for k in k_values:
            # Check if output file already exists
            output_dir = Path(base_output_dir) / dataset_name
            output_file = output_dir / f"{k}_nn.feather"
            
            if output_file.exists():
                logger.info(f"Skipping k={k} for {dataset_name} - file already exists: {output_file}")
                continue
            
            k_start_time = time.time()
            try:
                logger.info(f"Processing k={k} for {dataset_name}...")
                
                # Ensure output directory exists
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Fit KNN model
                nn_model = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
                nn_model.fit(training_X)
                
                # Find neighbors
                distances, indices = nn_model.kneighbors(testing_X)
                
                # Memory-efficient processing with chunking by test users
                logger.info(f"Processing neighbors in chunks by test users to manage memory...")
                
                # Initialize result accumulator
                all_results = []
                initial_chunk_size = 20     # Starting chunk size (changed from 5000 to 5)
                min_chunk_size = 5         # Minimum chunk size (changed from 100 to 1)
                chunk_size = initial_chunk_size
                n_test_users = len(indices)

                # Configuration for batch writing
                BATCH_SIZE = 10000  # Number of records per batch file
                batch_counter = 0
                current_batch_records = 0

                logger.info(f"Starting chunk processing with initial_chunk_size={initial_chunk_size}, min_chunk_size={min_chunk_size}")

                # Create output directory for batch files
                batch_dir = os.path.dirname(output_file)
                base_filename = os.path.splitext(os.path.basename(output_file))[0]
                batch_output_dir = os.path.join(batch_dir, f"{base_filename}_batches")
                os.makedirs(batch_output_dir, exist_ok=True)

                def write_batch_to_file(batch_data, batch_num):
                    """Write a batch of results to a separate file"""
                    if batch_data.empty:
                        return
                        
                    batch_filename = os.path.join(batch_output_dir, f"{base_filename}_batch_{batch_num:04d}.feather")
                    
                    # Merge with test_unseen for this batch
                    batch_merged = test_unseen.merge(batch_data, how='left', on=['item_id', 'user_id'])
                    
                    # Save batch file
                    batch_merged.to_feather(batch_filename)
                    logger.info(f"Batch {batch_num} saved to {batch_filename} with {len(batch_merged)} records")
                    
                    # Clean up
                    del batch_merged
                    gc.collect()

                def process_accumulated_results():
                    """Process and write accumulated results in batches"""
                    global batch_counter, current_batch_records
                    
                    if not all_results:
                        return
                    
                    # Combine accumulated results
                    combined_results = pd.concat(all_results, ignore_index=True)
                    logger.info(f"Combined {len(combined_results)} records from {len(all_results)} chunks")
                    
                    # Process in batches
                    total_records = len(combined_results)
                    batch_start = 0
                    
                    while batch_start < total_records:
                        batch_end = min(batch_start + BATCH_SIZE, total_records)
                        batch_data = combined_results.iloc[batch_start:batch_end].copy()
                        
                        write_batch_to_file(batch_data, batch_counter)
                        
                        batch_counter += 1
                        batch_start = batch_end
                        
                        # Clean up batch data
                        del batch_data
                        gc.collect()
                    
                    # Clear accumulated results
                    all_results.clear()
                    del combined_results
                    gc.collect()

                # Process test users in chunks with dynamic sizing
                chunk_start = 0
                while chunk_start < n_test_users:
                    try:
                        chunk_end = min(chunk_start + chunk_size, n_test_users)
                        
                        logger.debug(f"Processing chunk {chunk_start}-{chunk_end} with chunk_size={chunk_size}")
                        
                        # Create neighbors dataframe for this chunk of test users
                        chunk_test_users = []
                        chunk_neighbor_users = []
                        
                        for test_user_idx in range(chunk_start, chunk_end):
                            neighbor_indices = indices[test_user_idx]
                            chunk_test_users.extend([test_user_idx] * len(neighbor_indices))
                            chunk_neighbor_users.extend(neighbor_indices)
                        
                        if not chunk_test_users:  # Skip empty chunks
                            chunk_start = chunk_end
                            continue
                        
                        # Create neighbors dataframe for this chunk
                        chunk_neighbors_df = pd.DataFrame({
                            'test_user_id': chunk_test_users,
                            'neighbor_user_id': chunk_neighbor_users
                        })
                        
                        # Merge with training data
                        chunk_merged = chunk_neighbors_df.merge(
                            train_data, 
                            left_on='neighbor_user_id', 
                            right_on='user_id', 
                            how='inner'
                        )
                        
                        # Calculate average ratings for this chunk of test users
                        chunk_result = (chunk_merged.groupby(['test_user_id', 'item_id'])['rating']
                                    .mean()
                                    .reset_index()
                                    .rename(columns={'test_user_id': 'user_id', 'rating': 'avg_rating'}))
                        
                        all_results.append(chunk_result)
                        current_batch_records += len(chunk_result)
                        
                        # Clean up chunk data to free memory
                        del chunk_neighbors_df, chunk_merged, chunk_result
                        
                        # Check if we should write accumulated results to avoid memory buildup
                        if current_batch_records >= BATCH_SIZE * 2:  # Write when we have 2 batches worth
                            process_accumulated_results()
                            current_batch_records = 0
                        
                        # Successfully processed chunk, move to next
                        chunk_start = chunk_end
                        
                        # Reset chunk size to initial size after successful processing
                        # (in case it was reduced due to previous memory errors)
                        if chunk_size < initial_chunk_size:
                            chunk_size = min(initial_chunk_size, chunk_size + 5)  # Gradually increase back
                        
                        if chunk_end % 1000 == 0 or chunk_end == n_test_users:
                            logger.info(f"Processed {chunk_end}/{n_test_users} test users (chunk_size={chunk_size})")
                    
                    except MemoryError as e:
                        logger.error(f"MEMORY ERROR with chunk_size={chunk_size}: {str(e)}")
                        
                        # Clean up any partially created variables
                        if 'chunk_neighbors_df' in locals():
                            del chunk_neighbors_df
                        if 'chunk_merged' in locals():
                            del chunk_merged
                        if 'chunk_result' in locals():
                            del chunk_result
                        
                        # Force garbage collection
                        gc.collect()
                        
                        # Reduce chunk size
                        new_chunk_size = max(chunk_size - 5, min_chunk_size)
                        
                        if new_chunk_size < min_chunk_size:
                            logger.error(f"Chunk size would be below minimum ({min_chunk_size}). Cannot continue processing.")
                            raise RuntimeError(f"Memory error with minimum chunk size of {min_chunk_size}. Cannot process dataset {dataset_name} with k={k}")
                        
                        logger.warning(f"Reducing chunk_size from {chunk_size} to {new_chunk_size}")
                        chunk_size = new_chunk_size
                        
                        # Don't increment chunk_start - retry the same chunk with smaller size
                        continue

                # Process any remaining accumulated results
                if all_results:
                    process_accumulated_results()

                # Combine all batch files into final output file
                logger.info("Combining all batch files into final output...")
                batch_files = sorted([f for f in os.listdir(batch_output_dir) if f.endswith('.feather')])

                if batch_files:
                    final_results = []
                    
                    for batch_file in batch_files:
                        batch_path = os.path.join(batch_output_dir, batch_file)
                        batch_df = pd.read_feather(batch_path)
                        final_results.append(batch_df)
                        
                        # Clean up batch file after reading
                        os.remove(batch_path)
                        logger.debug(f"Processed and removed {batch_file}")
                    
                    # Combine all batches
                    k_avg_ratings = pd.concat(final_results, ignore_index=True)
                    
                    # Save final results
                    k_avg_ratings.to_feather(output_file)
                    logger.info(f"Final results saved to {output_file} with {len(k_avg_ratings)} records")
                    
                    # Clean up
                    del final_results, k_avg_ratings
                    
                    # Remove batch directory
                    os.rmdir(batch_output_dir)
                    logger.info("Batch directory cleaned up")
                    
                else:
                    # No results to combine - create empty result
                    k_avg_ratings = test_unseen.copy()
                    k_avg_ratings['avg_rating'] = None
                    k_avg_ratings.to_feather(output_file)
                    logger.warning("No results to combine - empty result dataframe created and saved")

                # Force garbage collection after each k value
                gc.collect()

                k_end_time = time.time()
                k_duration = k_end_time - k_start_time
                logger.info(f"k={k} completed in {k_duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"ERROR processing k={k} for {dataset_name}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        dataset_end_time = time.time()
        dataset_duration = dataset_end_time - dataset_start_time
        logger.info(f"Dataset {dataset_name} completed in {dataset_duration:.2f} seconds ({dataset_duration/60:.2f} minutes)")
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR processing dataset {dataset_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        dataset_end_time = time.time()
        dataset_duration = dataset_end_time - dataset_start_time
        logger.error(f"Dataset {dataset_name} failed after {dataset_duration:.2f} seconds")

def main():
        
    """
    Main function to process all datasets
    """
    total_start_time = time.time()

    # Define datasets
    datasets = {
        # 'bg': 'boardgamegeek.feather',
        'np': 'netflix_prize.feather',
        # 'm': 'movielens_25m.feather',
        # 'y': 'yahoo_r2_songs.subsampled.feather'
    }

    # Define k values to try
    k_values = [5, 10, 25, 50, 100, 150, 250, 350, 500]

    logger.info("Starting KNN processing for all datasets")
    logger.info(f"K values to process: {k_values}")
    logger.info(f"Total datasets: {len(datasets)}")
    logger.info("Note: Will skip processing if output file already exists")

    # Process each dataset
    for dataset_name, filename in datasets.items():
        try:
            logger.info(f"Loading dataset: {filename}")
            data = pd.read_feather(filename)
            logger.info(f"Dataset loaded successfully. Shape: {data.shape}")
            
            process_dataset(dataset_name, data, k_values)
            
            # Force garbage collection between datasets
            del data
            gc.collect()
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR loading dataset {filename}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            gc.collect()
            continue

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logger.info(f"{'='*60}")
    logger.info(f"ALL PROCESSING COMPLETED")
    logger.info(f"Total runtime: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes, {total_duration/3600:.2f} hours)")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()