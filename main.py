import argparse
from inspect_annotations import inspect_annotations
from inspect_metadata import inspect_metadata
from utils import download_dataset, build_sea_turtle_metadata, crop_turtle, display_cropped_images

parser = argparse.ArgumentParser(description='Federated Prototypical Network with Residual Networks and Atrous Spatial Pyramid Pooling for SeaTurtleID2022')
parser.add_argument("-ia", "--inspect-annotations", action="store_true", help="Inspect the dataset annotations") # Inspect the dataset annotations
parser.add_argument("-im", "--inspect-metadata", action="store_true", help="Inspect the dataset metadata")
parser.add_argument("-ov", "--overview", action="store_true", help="Display Project Overview") # Display the overview of the Thesis Project
parser.add_argument("-dd", "--download-dataset", action="store_true", help="Download the dataset")
parser.add_argument("-bd", "--build-data", action="store_true", help="Build Datasets")
parser.add_argument("-ct", "--category-print", action="store_true", help="Categorize and Display")

parser.add_argument("-mes", "--message", action="store_true", help="Message from the developers") # Small Easter Egg
parser.add_argument("-kie", "--kie-message", action="store_true", help="frankie's message") 


if __name__ == "__main__":
    args = parser.parse_args()

    # ================================================================
    # Descriptive Commands
    # ================================================================
    if args.overview:
        overview_msg = "Federated Prototypical Network with Residual Networks and Atrous Spatial Pyramid Pooling for SeaTurtleID2022"
        print("="*len(overview_msg))
        print("Project Overview: \n")
        print(overview_msg)
        print("="*len(overview_msg))

    if args.inspect_annotations:
        inspect_annotations()

    if args.inspect_metadata:
        inspect_metadata()

    # ================================================================
    # Utility Commands
    # ================================================================
    if args.download_dataset:
        download_dataset()


    # ================================================================
    # Miscellaneous commands
    # ================================================================
    if args.kie_message:
        print("u dont know how to update properly and you dont know how to clean.f")

    if args.message:
        developer_msg = "Nice to meet you! We are Nash Adam Muñoz and Elijah Kahlil Andres Abangan. Thank you for stopping by our Thesis Project!"
        print("="*len(developer_msg))
        print("Message from the Developers: \n")
        print(developer_msg)
        print("="*len(developer_msg))

    if args.build_data:
        annotations_path = "data/turtle-data/annotations.json"
        metadata_path = "data/turtle-data/metadata_splits.csv"
        dataset_path = "data"
        
        # Build the metadata files
        turtle_df, flipper_df, head_df = build_sea_turtle_metadata(
            annotations_path, 
            metadata_path, 
            dataset_path
        )
        
        print(f"\nTurtle metadata shape: {turtle_df.shape}")
        print(f"Flipper metadata shape: {flipper_df.shape}")
        print(f"Head metadata shape: {head_df.shape}")

    if args.category_print:
        turtle_id = "t001"
        metadata_path = "data/metadata_splits_turtle.csv"
        dataset_path = "data/turtle-data"
        
        print(f"Cropping images for turtle: {turtle_id}")
        print("=" * 60)
        
        # Crop turtle images
        cropped_images = crop_turtle(turtle_id, metadata_path, dataset_path)
        
        # Display results
        if cropped_images:
            print(f"\nDisplaying {len(cropped_images)} cropped images...")
            display_cropped_images(cropped_images, max_display=10)
            
            # Print details
            print("\nCropped Image Details:")
            for i, result in enumerate(cropped_images, 1):
                img_size = result['cropped_image'].size
                print(f"{i}. {result['file_name']}")
                print(f"   Category: {result['category']}")
                print(f"   Bounding Box: {result['bounding_box']}")
                print(f"   Cropped Size: {img_size[0]}x{img_size[1]}")
        else:
            print("No images were cropped.")
        
        # Example: Try different categories
        print("\n" + "=" * 60)
        print("You can also crop from other categories:")
        print("- metadata_splits_flipper.csv")
        print("- metadata_splits_head.csv")