from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from email.mime.base import MIMEBase
from email import*
from datetime import datetime
import time
import schedule
import pandas as pd
import numpy as np
msg = MIMEMultipart()
def atach(filename):
    
    fp = open(filename, 'rb')
    part = MIMEBase('application','vnd.ms-excel')
    part.set_payload(fp.read())
    fp.close()
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename=str(filename))
    msg.attach(part)   

def sendmail(text):
    print(text)
    df = pd.read_csv('historico.csv')
    for i in range(len(df.time)):
        df['time'].iloc[i] = df['time'].iloc[i].split('T')[0] 
    today = str(datetime.today()).split(' ')[0]
    mask = (pd.to_datetime(df['time']) >= today)
    df = df[mask]
    sell = df[df.op=='sell'].usd.sum()
    buy = df[df.op=='buy'].usd.sum()
     
    
    message = str(np.round(((buy-sell)/30)*100,3))+'% in '+str(len(df))+' operations'

    # setup the parameters of the message
    password = "darkruler1@"
    file1 = 'historico.csv'
    file2 = 'history.xlsx'
    msg['From'] = "guintherkovalski@gmail.com"
    msg['To'] = "guintherk14@gmail.com"
    msg['Cc'] = "guintherkovalski@gmail.com"
    msg['Subject'] = "Subscription"
    msg['Subject'] = 'test' 
    # add in the message body
    msg.attach(MIMEText(message, 'plain')) 
    #create server
    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()     
    atach(file1)
    atach(file2) 
    smtp = smtplib.SMTP('smtp.gmail.com') 
    # Login Credentials for sending the mail
    server.login(msg['From'], password)  
    # send the message via the server.
    server.sendmail(msg['From'], msg['To'], msg.as_string())  
    server.quit()
       

schedule.every().day.at("15:30").do(sendmail,'sending email')

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute
 

























