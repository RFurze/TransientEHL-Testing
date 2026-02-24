import argparse
import os

def extract_deformation_values(filepath):
    extracted_values = []
    current_last_deformation = None
    
    with open(filepath, 'r') as file:
        for line in file:
            # Capture the latest deformation value and overwrite the previous one
            if "Minimum film thickness um:" in line:
                value_str = line.split("Minimum film thickness um:")[1].strip()
                try:
                    current_last_deformation = float(value_str)
                except ValueError:
                    pass # Skip if the value isn't a valid float
                    
            # When we hit a Transient Step, save the last captured value
            elif "Transient Step =" in line:
                if current_last_deformation is not None:
                    extracted_values.append(current_last_deformation)
                    # Reset the variable for the next block
                    current_last_deformation = None
                    
    return extracted_values

if __name__ == "__main__":
    # Set up argument parsing to pass the path via the command line
    parser = argparse.ArgumentParser(description="Extract minimum film thickness from a log file.")
    parser.add_argument("filepath", type=str, help="Full directory path to the log file (e.g., /OneToOne/run.log)")
    
    args = parser.parse_args()
    
    # Verify that the file exists at the given path
    if not os.path.isfile(args.filepath):
        print(f"Error: Could not find the file at '{args.filepath}'. Please check the directory path.")
    else:
        # Run the extraction function
        values = extract_deformation_values(args.filepath)

        # Print the extracted values
        for idx, val in enumerate(values, start=1):
            print(f"Before Step {idx}: {val}")