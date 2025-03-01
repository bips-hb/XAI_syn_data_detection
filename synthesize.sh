echo "Downloading and saving data"
python data/get_data.py
echo "Downloading and saving data: Done"

echo "Synthesizing data"
for folder in data/*/; do
    dataset=${folder:5:-1}
    Rscript synthesize_non_dl.R $dataset ARF 10 16
    Rscript synthesize_non_dl.R $dataset synthpop 10 16
    python synthesize_dl.py --dataset $dataset --synthesizer TVAE --reps 10
    python synthesize_dl.py --dataset $dataset --synthesizer CTGAN --reps 10
    if [ $dataset != "diabetes_HI" ]
    then
        python synthesize_dl.py --dataset $dataset --synthesizer CTAB-GAN+ --reps 10
    else
        echo "Skipping"
    fi
    python synthesize_dl.py --dataset $dataset --synthesizer TabSyn --reps 10
done
echo "Synthesizing data: Done"

echo "Creating descriptive statistics"
Rscript data/create_descriptive.R
echo "Creating descriptive statistics: Done"