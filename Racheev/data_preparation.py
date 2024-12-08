import math
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Path parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


def to_categorical(car_table: pd.DataFrame):
    #car_table[df.select_dtypes('object').columns] = df.select_dtypes('object').astype('category')
    '''
    df.experience_level = pd.Categorical(df.experience_level)
    df = df.assign(experience_level=df.experience_level.cat.codes)
    df.employment_type = pd.Categorical(df.employment_type)
    df = df.assign(employment_type=df.employment_type.cat.codes)
    df.employee_residence = pd.Categorical(df.employee_residence)
    df = df.assign(employee_residence=df.employee_residence.cat.codes)
    df.company_size = pd.Categorical(df.company_size)
    df = df.assign(company_size=df.company_size.cat.codes)
    df.company_location = pd.Categorical(df.company_location)
    df = df.assign(company_location=df.company_location.cat.codes)
    df.job_title = pd.Categorical(df.job_title)
    df = df.assign(job_title=df.job_title.cat.codes)
    df.work_year = pd.Categorical(df.work_year)
    df = df.assign(work_year=df.work_year.cat.codes)
    df.remote_ratio = pd.Categorical(df.remote_ratio)
    df = df.assign(remote_ratio=df.remote_ratio.cat.codes)
    '''
    return car_table



def words(line, word_numb = 1, start_position = 0):
    words = line.split()
    return ' '.join(words[start_position : (start_position + word_numb)])

def clean_data(car_table_1: pd.DataFrame, car_table_2: pd.DataFrame, car_table_3: pd.DataFrame) -> pd.DataFrame:
    # Очистка и объединение данных

    # Удаление дублирующихся строк
    car_table_1.drop_duplicates(inplace=True, ignore_index=True)
    car_table_2.drop_duplicates(inplace=True, ignore_index=True)
    car_table_3.drop_duplicates(inplace=True, ignore_index=True)

    car_table_1 = car_table_1.rename(
        columns={'Car_Name': 'name', 'Year': 'year', 'Selling_Price': 'selling_price', 'Present_Price': 'present_price',
                 'Kms_Driven': 'km_driven', 'Fuel_Type': 'fuel', 'Seller_Type': 'seller_type',
                 'Transmission': 'transmission', 'Owner': 'owner'})

    # Удаление неиспользуемого столбца 'present_price'
    car_table_1 = car_table_1.drop('present_price', axis=1)
    #Преобразование стоимости в первой таблице к единому формату с остальными
    car_table_1['selling_price'] = round(car_table_1['selling_price']*math.pow(10,5))
    car_table_1['selling_price'] = car_table_1['selling_price'].astype(np.int64)
    #Приведие значений столбца 'owner' к единому формату данных (численный формат)
    uv_owner_t1 = car_table_1['owner'].unique()
    uv_owner_t2 = car_table_2['owner'].unique()
    uv_owner_t3 = car_table_3['owner'].unique()
    car_table_2['owner'] = car_table_2['owner'].replace(['Test Drive Car', 'First Owner', 'Second Owner',
     'Third Owner','Fourth & Above Owner'],['0','1','2','3','4'])
    car_table_3['owner'] = car_table_3['owner'].replace(['Test Drive Car', 'First Owner', 'Second Owner',
     'Third Owner','Fourth & Above Owner'],['0','1','2','3','4'])
    car_table_2['owner'] = car_table_2['owner'].astype(np.int64)
    car_table_3['owner'] = car_table_3['owner'].astype(np.int64)
    #Объединение таблиц
    car_table = pd.concat([car_table_1,car_table_2,car_table_3],axis = 0)
    #Удаление дублирующихся строк
    car_table.drop_duplicates(inplace=True, ignore_index=True)
    #Преобразование типа object в string
    car_table = car_table.convert_dtypes()
    #Удаление дублирования индексов
    car_table.reset_index(inplace=True)
    car_table.drop(columns='index', inplace = True)

    #Удаление выбросов цены
    car_table.drop(car_table[car_table['selling_price'] > 2000000 ].index, inplace=True)   #2000000  1200000
    car_table.drop(car_table[car_table['selling_price'] < 40000].index, inplace=True)    #40000  85000

    #Марка автомобилей обработка
    car_table['brand'] = car_table['name'].apply(words, word_numb=1)
    ## Преобразование марок автомобилей
    car_table['brand'] = car_table['brand'].replace(['Activa', 'Ambassador', 'Hero',        'Land', 'Maruti', 'OpelCorsa', 'Royal', 'alto'],
                                                    ['Honda', 'Hindustan', 'Dongfeng', 'Land Rover', 'Suzuki', 'Opel', 'Royal Enfield', 'Maruti'])
    car_table['brand'] = car_table['brand'].replace(['amaze', 'baleno', 'brio', 'camry', 'ciaz', 'city'],
                                                    ['Honda', 'Suzuki', 'Honda', 'Toyota', 'Suzuki', 'Honda'])
    car_table['brand'] = car_table['brand'].replace(
        ['corolla', 'creta', 'dzire', 'elantra', 'eon', 'ertiga', 'etios', 'fortuner'],
        ['Toyota', 'Hyundai', 'Suzuki', 'Hyundai', 'Hyundai', 'Suzuki', 'Toyota', 'Toyota'])
    car_table['brand'] = car_table['brand'].replace(['grand', 'i10', 'i20', 'ignis', 'innova', 'jazz', 'omni', 'ritz'],
                                                    ['Hyundai', 'Hyundai', 'Hyundai', 'Suzuki', 'Toyota', 'Honda', 'Suzuki', 'Suzuki'])
    car_table['brand'] = car_table['brand'].replace(['s',      'swift',    'sx4',  'verna',  'vitara',   'xcent', 'wagon'],
                                                    ['Suzuki', 'Maruti', 'Suzuki','Hyundai','Suzuki','Hyundai', 'Maruti'])
    ## Группировка марок
    car_table['brand'] = car_table['brand'].replace(['Maruti'], ['Suzuki'])
    car_table['brand'] = car_table['brand'].replace(
        ['Datsun', 'Ashok', 'Fiat', 'Chevrolet', 'UM', 'Hindustan', 'Hyosung', 'KTM', 'Royal Enfield', 'Opel', 'Daewoo',
         'Yamaha', 'Peugeot', 'Bajaj', 'TVS', 'Dongfeng'], 'Chevrolet')
    car_table['brand'] = car_table['brand'].replace(
        ['Lexus', 'Volvo', 'Land Rover', 'BMW', 'Jaguar', 'Mercedes-Benz', 'Audi'], 'BMW')
    car_table['brand'] = car_table['brand'].replace(
        ['MG', 'Jeep', 'Isuzu', 'Kia', 'Toyota', 'Mitsubishi', 'Force', 'Mahindra', 'Honda', 'Skoda', 'Ford',
         'Volkswagen', 'Nissan', 'Hyundai', 'Renault', 'Suzuki', 'Tata'], 'Suzuki')
    car_table['brand'] = car_table['brand'].replace(['BMW', 'Suzuki', 'Chevrolet'],
                                                    ['High class', 'Middle class', 'Low class'])
    # Преобразование года автомобиля
    car_table[(car_table['year'] <= 2000) & (car_table['selling_price'] > 300000)]
    # Удаление строк с электромобилем
    car_table = car_table.drop(car_table[car_table['fuel'] == 'Electric'].index)
    # Объёдинения газового топлива в одно
    car_table['fuel'] = car_table['fuel'].replace(['CNG', 'LPG'], 'Gaz')

    #Редактирование столбца пройденного расстояния
    car_table['km_dr_gr'] = pd.qcut(car_table['km_driven'], 4)
    qkm = car_table.groupby(['km_dr_gr'])['selling_price'].mean()
    qkm.reset_index()
    car_table['km_dr_gr'] = car_table['km_dr_gr'].astype(str)
    car_table['km_dr_gr'] = car_table['km_dr_gr'].replace(qkm.index.astype('str').to_list(), [35000,60000,100000,150000])
    car_table['km_dr_gr'] = car_table['km_dr_gr'].astype(np.int64)
                                                          # !!!!!!!Присвоим классам автомобилей численные значения для возможности обучения модели
    car_table['brand'] = car_table['brand'].replace(['High class', 'Middle class', 'Low class'], [2, 1, 0])
    car_table['km_dr_gr'] = car_table['km_dr_gr'].replace([35000, 60000, 100000, 150000], [0, 1, 2, 3])
    car_table['km_dr_gr'] = car_table['km_dr_gr'].astype('category')

    # Объёдинения диллеров в единых
    car_table['seller_type'] = car_table['seller_type'].replace(['Trustmark Dealer'], ['Dealer'])

    # Присвоим типам топлива автомобиля численные значения для возможности обучения модели
    car_table['fuel'] = car_table['fuel'].replace(['Petrol', 'Diesel', 'Gaz'], ['1', '0', '2']).astype(int)
    # Присвоим типам трансмиссии автомобиля численные значения для возможности обучения модели
    car_table['transmission'] = car_table['transmission'].replace(['Manual', 'Automatic'], ['0', '1']).astype(int)
    # Присвоим типам продавцов численные значения для возможности обучения модели
    car_table['seller_type'] = car_table['seller_type'].replace(['Dealer', 'Individual'], ['1', '0']).astype(int)
    car_table['km_dr_gr'] = car_table['km_dr_gr'].astype(int)

    #
    # Удалим столбец name, так как он имеет много различных текстовых данных, которые были обработаны в столбец brand
    car_table.drop(columns='name', inplace=True)
    # Удаление неиспользуемых столбцов
    car_table.drop(columns=['km_driven', 'seats'], inplace=True)
    car_table.drop(columns=['mileage', 'engine', 'max_power'], inplace=True)
    car_table.drop(columns='torque', inplace=True)

    car_table = to_categorical(car_table) #  примере в этом файле была функция
    return car_table

if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    i = 0
    src_table = []
    for data_file in input_dir.glob('*.csv'):
        src_table.append(data_file)
        #print(data_file)
        i = i + 1

        # Загрузка файлов
        '''
        src_table_1 = "car_data.csv"
        src_table_2 = "CAR_DETAILS_FROM_CAR_DEKHO.csv"
        src_table_3 = "Car_details_v3.csv"
        car_table_1 = pd.read_csv(src_table_1, delimiter=',')
        car_table_2 = pd.read_csv(src_table_2, delimiter=',')
        car_table_3 = pd.read_csv(src_table_3, delimiter=',')
        '''
        #car_table_clear = pd.read_csv('car_table_l_wp.csv', delimiter=',')  # требует указания в dvc
        #car_table = clean_data(car_table_1, car_table_2, car_table_3)
        #full_data = pd.read_csv(data_file)

        #!!!!
    car_table_1 = pd.read_csv(src_table[0], delimiter=',')
    car_table_2 = pd.read_csv(src_table[1], delimiter=',')
    car_table_3 = pd.read_csv(src_table[2], delimiter=',')
    #print(car_table_1.info(), car_table_2.info(), car_table_3.info())
    car_table_clear = clean_data(car_table_1, car_table_2, car_table_3)
    #print('Обработанное')
    #print(car_table_clear.info())
    X, y = car_table_clear.drop("selling_price", axis=1), car_table_clear['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=params.get('train_test_ratio'),
                                                        random_state=params.get('random_state'))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size=params.get('train_val_raitio'),
                                                      random_state=params.get('random_state'))
    X_full_name = output_dir / 'X_full.csv'
    y_full_name = output_dir / 'y_full.csv'
    X_train_name = output_dir / 'X_train.csv'
    y_train_name = output_dir / 'y_train.csv'
    X_test_name = output_dir / 'X_test.csv'
    y_test_name = output_dir / 'y_test.csv'
    X_val_name = output_dir / 'X_val.csv'
    y_val_name = output_dir / 'y_val.csv'

    X.to_csv(X_full_name, index=False)
    y.to_csv(y_full_name, index=False)
    X_train.to_csv(X_train_name, index=False)
    y_train.to_csv(y_train_name, index=False)
    X_test.to_csv(X_test_name, index=False)
    y_test.to_csv(y_test_name, index=False)
    X_val.to_csv(X_val_name, index=False)
    y_val.to_csv(y_val_name, index=False)

    #car_table.to_csv('car_table_l.csv', index=False)
