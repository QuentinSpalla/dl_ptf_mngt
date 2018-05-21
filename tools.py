
def rename_df(df, prefix='', suffix=''):
    """
    Changes column names of a dataframe by adding prefix and suffix
    :param df: pandas dataframe
    :param prefix: string,
    :param suffix: string
    :return: pandas dataframe, modified dataframe
    """
    df_colname = df.columns
    new_names = prefix + df_colname + suffix
    df.columns = new_names
    return df


def two_list_add(list1, list2, coeff):
    """
    Adds two lists element wise. Same size is mandatory
    :param list1: list
    :param list2: list
    :param coeff: float
    :return: list, modified list
    """
    list_pos = 0
    temp_list = [None] * len(list1)

    while list_pos < len(list1):
        temp_list[list_pos] = list1[list_pos] - coeff*list2[list_pos]
        list_pos += 1

    return temp_list