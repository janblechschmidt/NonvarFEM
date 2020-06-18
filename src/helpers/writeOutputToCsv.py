def writeOutputToCsv(csv_file, df):

    df.to_csv(csv_file, index=False)

    print("\nWrite output to", csv_file)
