#!/home/xfel/xfelopr/local/anaconda3/bin/python3


#	python check_Yometenai_Singo.py config_setting.xlsx config_sig.xlsx
#	python check_Yometenai_Singo.py config_setting.xlsx config_sig_test.xlsx

#	Normal
#	/home/xfel/xfelopr/local/anaconda3/bin/python3 -OO /home/xfel/xfelopr/kenichi/gtr/gtr.py /home/xfel/xfelopr/kenichi/gtr/config_XSBT_setting_SINGLE.xlsx /home/xfel/xfelopr/kenichi/gtr/config_XSBT_sig_SINGLE.xlsx 0

import time
import threading
from matplotlib import animation

import requests, bs4
from requests.exceptions import Timeout
import datetime
from datetime import datetime as dt
from datetime import timezone
import pytz
import re
import numpy as np
import matplotlib.pyplot as plt
import configparser
import time
import pandas as pd
import sys
import math
from matplotlib.dates import DateFormatter
from matplotlib.dates import num2date
from statistics import mean, median,variance,stdev
import queue
import pdb

#	for sound
import subprocess
#import winsound
#import seaborn as sns

plt.rcParams['font.family'] = "mikachan-PB"


print("arg len:",len(sys.argv))
if len(sys.argv) <= 3:
	print("Need arg")
print("argv:",sys.argv)
print("arg1:" + sys.argv[1])
conf_set = sys.argv[1]
conf_sig = sys.argv[2]



df_set = pd.read_excel(conf_set, sheet_name="setting", header=None, index_col=0)
print(df_set)
df_sig = pd.read_excel(conf_sig, sheet_name="sig")
#df_sig = pd.read_excel(conf_sig, sheet_name="sig", encoding='cp932')
#df_sig = pd.read_excel(conf_sig, sheet_name="sig", encoding='shift-jis')	#DAME
#df_sig = pd.read_excel(conf_sig, sheet_name="sig", encoding='utf-8')	#DAME
#df_sig = pd.read_html(conf_sig, sheet_name="sig")		#DAME
#print(df_sig)

#res = subprocess.run(["amixer", "sset", "Master", "on"], stdout=subprocess.PIPE)
#pdb.set_trace()


print("df_set.loc['interval']:	",df_set.loc['interval'])

if len(sys.argv) <= 7:
	strbegin = ""
	strend = ""
else:
	strbegin = sys.argv[7]
	strend = sys.argv[8]
	df_set.loc['interval'] = 100000000
print("begin:",strbegin)
print("end:",strend)

strbegin = strbegin.replace("+"," ")
strend = strend.replace("+"," ")
print("begin:",strbegin)
print("end:",strend)
print("df_set.loc['interval']:	",df_set.loc['interval'])






def is_nth_bit_set(num: int, n: int):
	if num & (1 << n):
		return True
	return False

def is_nth_bit_val(num: int, n: int):
	if num & (1 << n):
		return 1
	return 0

	

class SnaptoCursor(object):

    def __init__(self, ax):
        self.ax = ax
#        self.lx = ax.axhline(color='g')  # the horiz line
        self.ly = ax.axvline(color='g')  # the vert line
        #self.x = x
        #self.y = y
        # location of point
        
    def mouse_move(self, event):
        if not event.inaxes:
            return
        print('~~~~~~~~~~~~~~^')
        print(event)            
            
#        print('event.xdata = ' + str(event.xdata))
#        self.lx.set_ydata(-1000000000000000000)

        self.ly.set_xdata(event.xdata)
        self.ax.figure.canvas.draw()

#        if event.dblclick:
#        	print("double click")
#        	self.ly.set_xdata(event.xdata)
#        	self.ax.figure.canvas.draw()





def get_acc_sync(url):
    t = {}
    v = {}
#    print(url)
    try:
        res = requests.get(url, timeout=(30.0,30.0))   
    except Exception as e:
        print('Exception!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@get_acc_sync	' + url)
        print(e.args)
    else:
        res.raise_for_status()
        sp = res.text.split('<br>\n')        
        for line in sp:    
            m = re.search(r"(?P<tag>\d{1,10})(,)(\s)(?P<year>\d{4})(/)(?P<month>\d{1,2})(/)(?P<day>\d{1,2})(\s)(?P<Hour>\d{1,2})(:)(?P<minutes>\d{1,2})(:)(?P<sec>\d{1,2})(.)(?P<msec>\d{1,3})(,)(\s)(?P<val>.+)(,)", str(line))
            if m:
                t[int(m.group('tag'))] = datetime.datetime(int(m.group('year')),int(m.group('month')),int(m.group('day')),int(m.group('Hour')),int(m.group('minutes')),int(m.group('sec')),int(m.group('msec'))*1000)
                v[int(m.group('tag'))] = float(m.group('val'))
#                print(t)
#                print(v)
    finally:
	    return t,v
      


def get_data(url):
    t = {}
    v = {}
    sig_name = ""
    cnt = 0
    print(url)
    try:
        res = requests.get(url, timeout=(30.0,30.0))   
    except Exception as e:
        print('Exception!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@get_acc_sync	' + url)
        print(e.args)
    else:
        res.raise_for_status()
        sp = res.text.split('\n')        
        for line in sp:
#            print(line)
#            m = re.search(r"(?P<sig_id>\d{1,6})", str(line))
#           m = re.search(r"(?P<sig_id>\d{1,6})(&format=plot'>)(?P<sig_name>.+)(</a></td>)", str(line))
            m = re.search(r"(?P<sig_id>\d{1,6})(&format=plot'>)(?P<sig_name>.+)(</a>)", str(line))
            if m:
#                print("HIT")
#                print(m.group('sig_id'))
#                print(m.group('sig_name'))
                sig_name = m.group('sig_name')

            m2 = re.search(r"(<td ><font color=black>)(?P<val>.+)(</font>)(\s)", str(line))
            if m2:
#                print("HIT2")
#                print(m2.group('val'))
#                print(m2.group('val').find('8.88e+32'))
                if m2.group('val').find('8.88e+32') != -1:
                    print("YOMETENAI")
                    t[cnt] = sig_name
                    cnt=cnt+1
                if m2.group('val').find('0x7FFFFFFF') != -1:
                    print("YOMETENAI")
                    t[cnt] = sig_name
                    cnt=cnt+1
            
            m3 = re.search(r"(<td ><font color=>)(?P<val>.+)(</font>)(\s)", str(line))
            if m3:
#                print("HIT2")
#                print(m2.group('val'))
#                print(m2.group('val').find('8.88e+32'))
                if m3.group('val').find('8.88e+32') != -1:
                    print("YOMETENAI")
                    t[cnt] = sig_name
                    cnt=cnt+1
                if m3.group('val').find('0x7FFFFFFF') != -1:
                    print("YOMETENAI")
                    t[cnt] = sig_name
                    cnt=cnt+1
                    
                    

    finally:
        return t,v


    
    
    
class MyQ(queue.Queue):
    def show_value(self, i):
        print(self.queue[i])

    def sum(self):
        return sum(self.queue)

#    def ave(self):
#        return self.sum() / self.qsize()

    def ave(self):
        return np.mean(self.queue)

    def std(self):
    	return np.std(self.queue)
    	        

class SigInfo:
    def __init__(self, num):
        self.url = ''
        self.sta = ''
        self.sto = ''
        self.time = ''
        self.val = ''
        self.q = MyQ(num)
        self.rave = []
        self.rave_sigma = []
        self.t = {}
        self.d = {}        
        self.mu = 777
        self.sigma = 0
        self.flg_sound = 20
        
sig = [SigInfo(df_sig.loc[n]['rave']) for n in range(len(df_sig))]



sta = ''
flg_fist = 0
x_fix_flg= False
keys = []

"""
ax = [] * len(df_sig)
fig, (ax) = plt.subplots(nrows=len(df_sig), sharex="row", figsize=(float(x_size), float(y_size)))
fig.patch.set_facecolor(str(df_set.loc['bcolor']).replace("1","").strip().splitlines()[0])
fig.canvas.set_window_title(str(df_set.loc['title']).replace("1","").strip())

fig.canvas.manager.window.move(int(x_position), int(y_position))



for a in ax:
	a.patch.set_facecolor('gray')
	a.grid(axis="x", linestyle=':', color='snow')
	a.set_position([0.0,0.0,0.0,0])

xax_bar = 0.012
tes=[]
"""

#for n in range(len(df_sig)):
for n, s in enumerate(sig, 0):
	print("~~~~~~~~~~~~~~~~~~~~~~~~label:	" + df_sig.loc[n]['label'])
	s.time, s.val = get_data(df_sig.loc[n]['sid'])
	print(s.time)
	
#	if  df_sig.loc[n]['ax'] == 1:
#	    tes.append(Tesclick(ax[n]))	    
#	    ax[n].set_position([0, (max(df_sig.loc[:]['graph'])-df_sig.loc[n]['graph'])*(1/(max(df_sig.loc[:]['graph'])+1))+xax_bar, 1, 1/(max(df_sig.loc[:]['graph'])+1)])
#	    ax[n].patch.set_facecolor( df_sig.loc[n]['fcolor'] )
	    



"""----------------------------------------------------------------------------------

#			print("keys:	" + str(len(keys)))

#		print(sig[0].d.values())
		
		
		print('### Updated ###  ' + sta.strftime("%Y/%m/%d+%H:%M:%S") + ' ~ ' + sto.strftime("%Y/%m/%d+%H:%M:%S") + '  len(keys): ' +  str(len(keys)) )
		time.sleep(int(df_set.loc['interval']))






def _redraw(_):
	global flg_fist
	global x_min_init
	global x_fix_flg
	global x_min
	global x_max
	

#	plt.show()		#????

#	if not keys:
#		print('DEBUG@<<< redraw >>>		No keys[] '
#		return


	print(keys)
		
	if not keys:
		print('Debug@<<< redraw >>>	No key')
		return

	print('Debug@<<< redraw >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>	')	
	if flg_fist == 0 and keys:
#		print(keys)
		sorted_list = sorted(keys)
#		print(sorted_list)
		x_min = sorted_list[0]
		x_min_init = x_min


	if not 'x_min' in globals():
		return
		
	if not x_fix_flg and keys:	# Full SPAN
		x_min = x_min_init
		x_max = keys[-1]

	if not 'x_max' in globals():
		return
			
	for n, s in enumerate(sig, 0):
		ax[n].set_xlim([x_min, x_max + (x_max -x_min)/12])

#	if not keys:
#		return
		

	df_sig = pd.read_excel(conf_sig, sheet_name="sig")
	df_set = pd.read_excel(conf_set, sheet_name="setting", header=None, index_col=0)			
	
    
	for n, s in enumerate(sig, 0):
		if len(keys) != len(s.rave):
			print(str(df_sig.loc[n]['sname']) +	"	Not match dimension	keys:" + str(len(keys))	+ "		s.rave" + str(len(s.rave)))
			return
int( pd.pivot_table(df, index=[df_sig.loc[0]['label']], columns=[df_sig.loc[1]['label']] ) )
	
#	print(keys)

	print('<<< redraw >>>	' + str(x_min) + '	- 	' + str(x_max))	
#	print(sig[0].time)
	for n, s in enumerate(sig, 0):
#		print(n)
#		print(keys)
#		print(s.url)
#		print(s.time)
#		print(s.rave)
#		print(s.rave_sigma)
#		print(s.mu)
#		print(s.sigma)
		
		if s.mu!=777:
			y_range = s.sigma*float(df_sig.loc[n]['y_range'])*3 if int(df_sig.loc[n]['rave'])!=1 else float(df_sig.loc[n]['y_range'])*3
			y_tolerance = s.sigma*float(df_sig.loc[n]['y_range']) if int(df_sig.loc[n]['rave'])!=1 else float(df_sig.loc[n]['y_range'])

#	MOTO
#			y_range = s.sigma*float(df_set.loc['y_range']) if pd.isnull( df_sig.loc[n]['y_range'] ) else float(df_sig.loc[n]['y_range'])
#			y_tolerance = s.sigma*float(df_set.loc['y_tolerance']) if pd.isnull( df_sig.loc[n]['y_range'] ) else float(df_sig.loc[n]['y_range'])*0.3
			
			ax[n].set_ylim(s.mu-y_range, s.mu+y_range)
			if df_sig.loc[n]['ax']==1:
				ax[n].axhspan(s.mu-y_tolerance, s.mu+y_tolerance,	color='#333333')
					
			ax[n].axhline(s.mu, color=str(df_set.loc['bcolor']).replace("1","").strip().splitlines()[0], linewidth=0.1)
			if not pd.isnull( df_sig.loc[n]['sound']):
				ax[n].axhline(s.mu*float(df_sig.loc[n]['sound']), color="crimson", linewidth=0.3, linestyle="dashed")

	
		if int(df_sig.loc[n]['calc-sigma'])==0:
			if int(df_sig.loc[n]['rave'])==1:
				ax[n].plot(keys, s.rave, linestyle="solid", marker=str(df_sig.loc[n]['marker']).replace("1","").strip().splitlines()[0], markersize=df_sig.loc[n]['linewidth'], color=df_sig.loc[n]['color'], label=df_sig.loc[n]['label'], clip_on=False)
			else:
				ax[n].plot(keys, s.rave, linestyle="solid", linewidth=df_sig.loc[n]['linewidth'], color=df_sig.loc[n]['color'], label=df_sig.loc[n]['label'], clip_on=False)
		else:
			ax[n].errorbar(keys, s.rave, yerr = s.rave_sigma, elinewidth=0.001, capsize=1, capthick=0.3, alpha=0.5, marker=str(df_sig.loc[n]['marker']).replace("1","").strip().splitlines()[0], markersize=df_sig.loc[n]['linewidth'], linestyle="solid", color=df_sig.loc[n]['color'], label=df_sig.loc[n]['label'])

#		ax[n].set_xlim([x_min, x_max + (x_max -x_min)/12])

#		ax[n].gca().spines['top'].set_visible(False)
#		ax[n].gca().spines['bottom'].set_visible(False)
		
		if flg_fist == 0:
			ax[n].legend(loc='upper left', bbox_to_anchor=(0.2*(df_sig.loc[n]['ax']-1), 0, 0.35, 0.3))
#			ax[n].legend(loc='upper left', bbox_to_anchor=(0.12*(df_sig.loc[n]['ax']-1), 0, 0.35, 0.3))				MOTO
#			ax[n].xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
#			ax[n].xaxis.set_major_formatter(DateFormatter('%d %H:%M'))
			ax[n].xaxis.set_major_formatter(DateFormatter('%-m/%-d(%-H:%-M)'))
#			ax[n].xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
			ax[n].tick_params(axis='x', labelsize=14)

#	keys.clear()
	flg_fist = 1

def _redraw2(_):
	if not keys:
		return

	df_sig = pd.read_excel(conf_sig, sheet_name="sig")
	df_set = pd.read_excel(conf_set, sheet_name="setting", header=None, index_col=0)

	for n, s in enumerate(sig, 0):
		if len(keys) != len(s.rave):
			print(str(df_sig.loc[n]['sname']) +	"	Not match dimension	keys:" + str(len(keys))	+ "		s.rave" + str(len(s.rave)))
			return

		
	for n, s in enumerate(sig, 0):
		if s.mu!=0:
			sx[n].hist(sig[n].rave, bins=20, alpha=0.65, label="Haiki", color="red", stacked=False)



def _init():
	t = threading.Thread(target=_update)
	t.daemon = True
	t.start()

params = {
	'fig': fig,
	'func': _redraw,  # グラフを更新する関数
	'init_func': _init,  # グラフ初期化用の関数 (データ更新用スレッドの起動)
#	'fargs': keys,  # 関数の引数 (フレーム番号を除く)
	'interval': 13 * 1000,  # グラフを更新する間隔 (ミリ秒)
}
anime = animation.FuncAnimation(**params)


params2 = {
	'fig': fig2,
	'func': _redraw2,  # グラフを更新する関数
	'init_func': _init,  # グラフ初期化用の関数 (データ更新用スレッドの起動)
#	'fargs': keys,  # 関数の引数 (フレーム番号を除く)
	'interval': 53 * 1000,  # グラフを更新する間隔 (ミリ秒)
}
anime2 = animation.FuncAnimation(**params2)


plt.show()
"""