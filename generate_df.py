import numpy as np 
import pandas as pd 

number_questions = int(input('Please write the number of questions: '))
number_options = int(input('Please write the number of options: '))

abc = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
def change_num(number_options, abc):
    options =  [abc[i] for i in range(number_options)]
    list_options = [i for i in range(number_options)]
    return options, list_options
options, list_options = change_num(number_options, abc)

def creation_test(number_questions, number_options):
    df = { i:[ 'O' for _ in range(number_options)] for i in range(number_questions)}
    df = pd.DataFrame(df)
    df.index = options
    return df
df = creation_test(number_questions, number_options)

def mod(options, df):
    for num_q in df:
        correct_option = input(f'Please write the correct option for question {num_q}: ').upper()
        option_index_map = {option: i for i, option in enumerate(options)}

        while correct_option not in options:
            print('Please write a correct option')
            correct_option = input(f'Please write the correct option for question {num_q}: ').upper()

        if correct_option in option_index_map:
            idx = option_index_map[correct_option]
            df.iloc[idx, num_q] = 'X'

    return df

modi = mod(options, df)
print(modi)