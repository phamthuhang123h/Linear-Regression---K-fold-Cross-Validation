#Bài tiểu luận số 1
#Nhóm 9
#Các thành viên:
    #Nguyễn Duy Anh: 2051060385
    #Phạm Thu Hằng: 2051063867
    #Phan Trung Hiếu: 2051063732
#Giảng viên hướng dẫn: TS. Nguyễn Thị Kim Ngân

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression #hồi quy tuyến tính
from sklearn.model_selection import train_test_split, KFold
import numpy as np


data = pd.read_csv('./diamonds.csv')

#Tạo vector với các giá trị tương ứng các thuộc tính
x_data = np.array (data[["carat", "cut", "color", "clarity", "depth","price"]].values)

#Định nghĩa hàm convert giá trị từ dạng string sang numeric để thực hiện phép tính
def data_encoder(x):
    for i,j in enumerate(x):
        for k in range(0,6): 
            if (j[k] == "Ideal"):
                j[k] = 0
            elif (j[k] == "Premium"):
                j[k] = 1
            elif (j[k] == "Very Good"):
                j[k] = 2
            elif (j[k] == "Good"):
                j[k] = 3
            elif (j[k] == "Fair"):
                j[k] = 4
            elif (j[k] == "D"):
                j[k] = 5
            elif (j[k] == "E"):
                j[k] = 6
            elif (j[k] == "F"):
                j[k] = 7
            elif (j[k] == "G"):
                j[k] = 8
            elif (j[k] == "H"):
                j[k] = 9
            elif (j[k] == "I"):
                j[k] = 10
            elif (j[k] == "J"):
                j[k] = 11
            elif (j[k]=="IF"):
                j[k] = 12
            elif (j[k]=="VVS1"):
                j[k] = 13
            elif(j[k]=="VVS2"):
                j[k] = 14
            elif(j[k]=="VS1"):
                j[k] = 15
            elif (j[k]=="VS2"):
                j[k] = 16
            elif(j[k]=="SI1"):
                j[k] = 17
            elif(j[k]=="SI2"):
                j[k] = 18
            elif(j[k]=="I1"):
                j[k] = 19
    return x
#Chuyển dữ liệu sử dụng hàm  encoder vừa định nghĩa
data_encoder_new = data_encoder(x_data)

print("data encode: ")
print(data_encoder_new)


#Chia tập dữ liệu thành 7 phần huấn luyện, 3 phần kiểm thử
dt_Train, dt_Test = train_test_split(data_encoder_new, test_size = 0.3, shuffle = False)


#Chuyển về dataframe của pandas để sử dụng hàm iloc
dt_Train = pd.DataFrame(dt_Train)
dt_Test = pd.DataFrame(dt_Test)

#Định nghĩa hàm tìm sai số
def error(y, y_pred):
    l = [] #tao mang l
    for i in range(0, len(y)): 
        l.append(np.abs(np.array(y[i:i+1]) - np.array(y_pred[i:i+1]))) #price thực tế - price dự đoán
    return np.mean(l) #Tính trung bình tất cả các giá trị sai số

#Chia tập dữ liệu làm k phần, ở đây k =410, lấy theo thứ tự, ko random
k=410 #410 vì đây là 70% của tập dữ liệu
kf = KFold(n_splits = k, random_state = None) #Chia k phan de train
max = 9999999 #Lấy số lớn hẳn nhiều
i = 1
for train_index, test_index in kf.split(dt_Train):
    #Đọc dữ liệu
    X_train, X_test = dt_Train.iloc[train_index, :5], dt_Train.iloc[test_index, :5]
    y_train, y_test = dt_Train.iloc[train_index, 5], dt_Train.iloc[test_index, 5]

    #Hồi quy tuyến tính
    lr = LinearRegression() #Tạo hàm hồi quy tuyến tính
    lr.fit(X_train, y_train) #Đưa các dữ liệu trong tập huấn luyên X_train và tập nhãn lớp Y_Train vào
    y_pred_train = lr.predict(X_train) #Dự đoán giá price dựa trên tập huấn luyện
    y_pred_test = lr.predict(X_test) #Dự đoán giá price dựa trên tập kiểm thử

    #Tính tổng giá trị trung bình sai số (price thực tế - price dự đoán) của tập huấn luyện và kiểm thử
    sum = error(y_train, y_pred_train) + error(y_test, y_pred_test) 


    #Lấy giá trị sum nhỏ nhất, lặp cho i tăng từ i=1 ở trên
    if sum < max:
        max = sum
        last = i
        regr = lr.fit(X_train, y_train)
    i = i + 1
y_predict = regr.predict(dt_Test.iloc[:, :5]) #DU doan gia tien y cua mo hinh hoc may tot nhat
y = np.array(dt_Test.iloc[:, 5]) #lay gia tri tien thuc te cua tap test
#coefficient of determination là R^2: Tức trọng số sao cho giá trị (price thực tế - price dự đoán) là nhỏ nhất
print("Coefficient of determination: %.2f" %error(y_test, y_predict))
print("Thực tế \t Dự đoán \t Chênh lệch")
for i in range (0, len(y)):
    print("%.2f" % y[i], " ", y_predict[i], " ", abs(y[i] - y_predict[i]))

#Hàm này tính tỉ lệ đúng
def percent_true():
    count = 0
    for i in range(0, len(y)):
        if abs(y[i] - y_predict[i]) <= error(y_test, y_predict):
               count = count + 1
    return (count/len(y)) * 100
print("ty le dung: ", percent_true())
#Tỉ lệ sai = 100 - tỉ lệ đúng
print("ty le sai: ", 100 - percent_true())
