

# first input is the document name we want to pull and plot
input_csv="$1"

# check if input is provided
if [ -z "$input_csv" ]; then
    echo "Usage: $0 <input_csv_filename>"
    exit 1
fi

adb pull "/data/local/tmp/hexagon_test/${input_csv}" "./${input_csv}"
python3 plot_csv.py "./${input_csv}" "${input_csv%.csv}.png"