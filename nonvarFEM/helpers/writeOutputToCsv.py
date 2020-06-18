import os


def writeOutputToCsv(df, opt):
    """ Function to write the data frame to a csv-file.
    """

    os.makedirs(opt['outputDir'], exist_ok=True)
    thisPath = os.path.join(opt['outputDir'], opt['id'])
    csvFile = thisPath + '.csv'
    df.to_csv(csvFile, index=False)

    print("\nOutput written to {}".format(csvFile))
