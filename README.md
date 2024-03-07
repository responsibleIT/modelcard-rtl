# AAI_RTL_2022



## Getting started

In this file you will be provided with an explanation of how to use and reproduce the model card on your own machine and with your own database. The necessary files are all found in the folder named dashboard.

## Install the necessary requirements 

Install the requirements.txt file by running the command seen below. This will provide you with the necessary packages.

```
cd "Directory of the folder"
pip install -r requirements.txt
```

## Prepare the data

When using the dashboard you should be able to run it right from the start. The data is located in the OEGE database from the HvA and the engine within the file connects to that database.

If you want to replace this database with your own, you should remake the engine to fit your credentials. You should also import the 3 CSV files into your database. **IMPORTANT** make sure the tables are named correctly, directly importing them and copy and pasting the name from the CSV works and is the preffered method. If the tables are given another name this needs to be changed in te code ass well.

## Run the file

Run the file. You will be provided with a localhost link. When this link is put into the browser de dashboard will appear and you are free to use it ass you see fit.


## Extra information

- All the filters within the dashboard are made to fit the existing column names, don't change these.
- The LM score filter and the per sentence filter work by searching for a file within the database. All the available files are visible and autocomplete works ass well. If a file isn't seen within the autocomplete table it doesn't exist.

