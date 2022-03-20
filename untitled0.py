# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:50:04 2022

@author: RRZ
"""
heroku git:remote -a example-app

git remote add origin git@github.com:RichardRuiHuZhang/arcane-escarpment-09352.git

https://git.heroku.com/arcane-escarpment-09352.git

git remote add origin git@github.com:RichardRuiHuZhang/apitest01.git


web: uvicorn main:app --host "0.0.0.0" --port ${PORT}


web: gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app