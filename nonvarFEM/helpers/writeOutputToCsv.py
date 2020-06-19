import os


def writeOutputToCsv(df, opt, fname=None):
    """ Function to write the data frame to a csv-file.
    """

    if fname is None:
        fname = opt['id']

    os.makedirs(opt['outputDir'], exist_ok=True)
    thisPath = os.path.join(opt['outputDir'], fname)
    csvFile = thisPath + '.csv'
    df.to_csv(csvFile, index=False)

    print("\nOutput written to {}".format(csvFile))
