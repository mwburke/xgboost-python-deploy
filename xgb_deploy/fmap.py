import numpy as np


def generate_fmap(feature_names, feature_types, fout='fmap.txt'):
    """
    Generates fmap file to be used in model export so that the
    model can retain  metadata for names and types of data it
    can expect in prediction.

    Args:
        feature_names: List of names of all features of the data
            in the order they are found in the dMatrix.
        feature_types: List of types of data for each feature in
            feature_names. Available options are:
            i: binary indicator feature
            q: quantitative value
            int: integer value with more than two options
        fout: Name of the output file to write the fmap to.
            Defaults to 'fmap.txt' in the current directory.
    """
    with open(fout, 'w') as f:
        for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
            if ftype not in ['i', 'int', 'q']:
                raise ValueError("Please limit feature types to 'i', 'int', or 'q'")
            f.write('{} {} {}\n'.format(i, name, ftype))


def generate_fmap_from_pandas(df, fout='fmap.txt'):
    """
    Genereates fmap file from a pandas dataframe. Checks column
    types for ints or floats, and infers types from there.

    If there is a colum dtype that does not correspond to either
    an integer or float, it throws an error and asks the user to
    review.

    If column names have spaces in them, it throws an error and
    asks the user to review.

    Args:
        df: Pandas DataFrame with data to be input directly into
            XGBoost sklearn-like API or into a dMatrix.
        fout: Name of the output file to write the fmap to.
            Defaults to 'fmap.txt' in the current directory.
    """
    feature_names = df.columns
    feature_types = []
    feature_dtypes = df.dtypes
    for name in feature_names:
        if ' ' in name:
            raise ValueError('Please change column {} so that it contains no whitespace.'.format(name))
        if feature_dtypes[name] in [np.dtype('float64'), np.dtype('float32'), np.dtype('float16')]:
            type = 'q'
        elif feature_dtypes[name] in [np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8')]:
            if np.array_equal(df[name].unique(), np.array([0, 1])):
                type = 'i'
            else:
                type = 'int'
        else:
            raise ValueError('Please review column {} as it is neither a float nor an int'.format(name))
        feature_types.append(type)

    generate_fmap(feature_names, feature_types, fout)
