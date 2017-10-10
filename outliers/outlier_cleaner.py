#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    errors = predictions - net_worths

    mydata = []
    for i in range(len(errors)):
        mytuple = ages[i], net_worths[i], errors[i]
        mydata.append(mytuple)


    cleaned_data = sorted(mydata, key = lambda x: x[2])[:-9]

    ### your code goes here
    return cleaned_data
