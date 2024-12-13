/* this script experiments with univariate rnn / nnet for time series */

/* load data */
data rnn_tst;
set '/export/home/users/sukhsn/RNNs/fpp2_sunspotarea.sas7bdat';
run;

/* upload into cas */
data casuser.rnn_tst;
set rnn_tst;
run;

/* prepare data with 10 lags */
proc tsmodel data = casuser.rnn_tst outarray= casuser.test;
	id year interval = year start='01jan1875'd end='30dec2025'd setmiss=missing;
	outarrays var1-var10;
	var value;
	submit;
		do i = 1 to dim(value);
			if i>1 then var1[i]=value[i-1];
			if i>2 then var2[i]=value[i-2];
			if i>3 then var3[i]=value[i-3];
			if i>4 then var4[i]=value[i-4];
			if i>5 then var5[i]=value[i-5];
			if i>6 then var6[i]=value[i-6];
			if i>7 then var7[i]=value[i-7];
			if i>8 then var8[i]=value[i-8];
			if i>9 then var9[i]=value[i-9];
			if i>10 then var10[i]=value[i-10];
		end;
	endsubmit;
run;


/* creating training portion */
data casuser.train;
	set casuser.test;
	if year(year)<=2015 then output;
run;



/********************************************/
/*  										*/
/*  		Model 1: PROC NNET				*/
/*  										*/
/********************************************/

/* running the neural net procedure with 6 nodes in the hidden layer */
/* the score code is generated in a sas file. The code will be used to score new data afterwards */
proc nnet data=casuser.train;
	input var1-var10;
	target value /level=int;
	hidden 6;
	train outmodel=casuser.nnetmodel seed=12345;
	code file='/export/home/users/sukhsn/RNNs/model.sas';
run;

/* The score code is used to predict the future. Lagged variables are computed using the retain statement */
data casuser.fit;
	set casuser.test;
	retain p_value copyvar1-copyvar10;
	if var10=. then var10 = copyvar9;
	if var9=. then var9 = copyvar8;
	if var8=. then var8 = copyvar7;
	if var7=. then var7 = copyvar6;
	if var6=. then var6 = copyvar5;
	if var5=. then var5 = copyvar4;		
	if var4=. then var4 = copyvar3;
	if var3=. then var3 = copyvar2;
	if var2=. then var2 = copyvar1;
	if var1=. then var1 = p_value;
	
	/* score code is put as below */
	%include '/export/home/users/sukhsn/RNNs/model.sas';
	
	/* a copy of the current observation is saved to create lagged variables in the next iteration */
	copyvar1 = var1;
	copyvar2 = var2;
	copyvar3 = var3;
	copyvar4 = var4;
	copyvar5 = var5;
	copyvar6 = var6;
	copyvar7 = var7;
	copyvar8 = var8;
	copyvar9 = var9;
	copyvar10 = var10;
run;


/* plot predictions */
proc sgplot data=casuser.fit;
series x=year y=value / ;
series x=year y=p_value;
title 'Actual v Forecast';
run;

/* calculate fit statistics */
data fit;
set casuser.fit;
where year(year) ge 1900 and year(year) le 2015;
sse = (value - p_value)**2;
keep year value p_value sse;
run;

proc means data=fit; var sse;run;

proc sql;
 select sqrt(mean((value-p_value)**2)) as rmse from casuser.fit;quit;


/********************************************/
/*  										*/
/*  		Model 2: deepLearn				*/
/*  										*/
/********************************************/

/* define model */
proc cas;
session casauto;

deepLearn.buildModel / 
modelTable = {name='test_rnn',replace=TRUE} type="RNN";run;

deepLearn.addlayer / layer = {type="INPUT",std="STD"}  modelTable = {name='test_rnn'} name='data';run;
deepLearn.addlayer / layer={type='recurrent' n=64 rnnType='lstm',outputtype='samelength'}      modelTable={name="test_rnn"}      
      name="rnn1"
      srcLayers={"data"};run;

deepLearn.addlayer / layer={type='recurrent' n=64
             rnnType='lstm',outputtype='samelength'}      modelTable={name="test_rnn"}      
      name="rnn2"
      srcLayers={"rnn1"};run;

deepLearn.addlayer / layer={type='recurrent' n=32
             rnnType='lstm',outputtype='encoding'}      modelTable={name="test_rnn"}      
      name="rnn3"
      srcLayers={"rnn2"};run;

deepLearn.addlayer / layer={type='output'
            }      modelTable={name="test_rnn"}      
      name="output"
      srcLayers={"rnn3"};run;



deepLearn.modelinfo / modelTable={name="test_rnn"};run;
quit;


/* train model */
proc cas;
session casauto;
deepLearn.dlTrain / inputs={"var1","var2","var3","var4","var5","var6","var7","var8","var9","var10"} missing='NONE' modeltable = {name="test_rnn"} modelweights={name="test_rnn_weights",replace=TRUE}
table={name="train",caslib="casuser"}
nThreads=1
optimizer={algorithm={method="ADAM",learningrate=0.005,learningratepolicy='STEP'}, maxepochs=100}
target="value";run;
quit;


/* model inference */
proc cas;
session casauto;
   deepLearn.dlScore /                                                    /*2*/
      casOut={name="test_scored", replace=TRUE} copyvars={"year"}
      initWeights={name="test_rnn_weights"}
      modelTable={name="test_rnn"}
      table={name="test"} ;
run;

proc cas;
session casauto;
table.fetch / table={name='test_scored'};run;
quit;


data casuser.viz_preds;
merge casuser.test_scored casuser.test;
by year;
run;

proc sgplot data=casuser.viz_preds;
series x=year y=value / ;
series x=year y=_DL_Pred_;
title 'Actual v Forecast';
run;

/* this doesn't fit as there's <150 records in the data */

/********************************************/
/*  										*/
/*  	Model 3: RNN with more data			*/
/*  										*/
/********************************************/

/* simulate a time series with seasonality and noise */
proc python;
submit;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)

T = 20
L = 2000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / T).astype('float64')

fig, ax = plt.subplots(1,1,figsize=(10,3))
plt = ax.plot(data[0, :])

SAS.pyplot(fig)

dfout = pd.DataFrame({'y':list(data[0,:])}).reset_index()

SAS.df2sd(dfout,'work.dfout')
endsubmit;
quit;

data casuser.sin;
set work.dfout;
date = intnx('day','01JAN95'd,index);
y_tx = y + rand('normal');
format date date9.;
run;

/* prepare the time series */
proc tsmodel data = casuser.sin outarray= casuser.sin_tst;
	id date interval = day start='01jan1995'd end='02jul2000'd setmiss=missing;
	outarrays var1-var10;
	var y_tx;
	submit;
		do i = 1 to dim(y_tx);
			if i>1 then var1[i]=y_tx[i-1];
			if i>2 then var2[i]=y_tx[i-2];
			if i>3 then var3[i]=y_tx[i-3];
			if i>4 then var4[i]=y_tx[i-4];
			if i>5 then var5[i]=y_tx[i-5];
			if i>6 then var6[i]=y_tx[i-6];
			if i>7 then var7[i]=y_tx[i-7];
			if i>8 then var8[i]=y_tx[i-8];
			if i>9 then var9[i]=y_tx[i-9];
			if i>10 then var10[i]=y_tx[i-10];
		end;
	endsubmit;
run;

/* create training dataset */
data casuser.sin_train;
	set casuser.sin_tst;
	if year(date)<2000 then output;
run;

/* train model */
proc cas;
session casauto;

deepLearn.buildModel / 
modelTable = {name='test_rnn',replace=TRUE} type="RNN";run;

deepLearn.addlayer / layer = {type="INPUT",std="STD"}  modelTable = {name='test_rnn'} name='data';run;
deepLearn.addlayer / layer={type='recurrent' n=15 rnnType='lstm',outputtype='samelength'}      modelTable={name="test_rnn"}      
      name="rnn1"
      srcLayers={"data"};run;

deepLearn.addlayer / layer={type='recurrent' n=15
             rnnType='lstm',outputtype='encoding'}      modelTable={name="test_rnn"}      
      name="rnn2"
      srcLayers={"rnn1"};run;


deepLearn.addlayer / layer={type='output' act='IDENTITY' }      modelTable={name="test_rnn"}      
      name="output"
      srcLayers={"rnn2"};run;



deepLearn.modelinfo / modelTable={name="test_rnn"};run;
quit;


/* train model */
proc cas;
session casauto;
deepLearn.dlTrain / inputs={"var1","var2","var3","var4","var5","var6","var7","var8","var9","var10"} modeltable = {name="test_rnn"} modelweights={name="test_rnn_weights",replace=TRUE}
table={name="sin_train",caslib="casuser"}
nThreads=1
optimizer={algorithm={method="ADAM",learningrate=0.001}, maxepochs=100}
target="y_tx";run;
quit;

/* ignore for now... */
data casuser.testset;
set casuser.sin_tst;
if date lt '30jun2000'd then output;
run;



/* model inference */
proc cas;
session casauto;
   deepLearn.dlScore /                                                    /*2*/
      casOut={name="test_scored", replace=TRUE} copyvars={"date"} 
      initWeights={name="test_rnn_weights"}
 layerOut={name="layer", replace=TRUE}
      modelTable={name="test_rnn"}
      table={name="sin_tst",vars={"y_tx"}} topprobs=5;
run;

proc cas;
session casauto;
table.fetch / table={name='test_scored'};run;
quit;


data casuser.viz_preds;
merge casuser.test_scored casuser.sin_tst;
by date;
run;

proc sgplot data=casuser.viz_preds;
series x=date y=y_tx / ;
series x=date y=_DL_Pred_;
title 'Actual v Forecast';
where year(date) ge 1999;
run;

/* get current forecast date */
proc fedsql sessref=casauto;
select max(date) from casuser.viz_preds where _DL_Pred_ <> . ; quit;


/* macro for multi-step forecast

- inputs: model table, series variable, output table, copyvars,  name of pred var, number of steps?

logic
- score input table to create initial _DL_Pred_ variable
- create loop for N-1 steps using VARS=_DL_Pred_
- create output table with join 
 */


/* for some reason we only get a 1 step forecast here... */

/********************************************/
/*  										*/
/*  		Model 3: TSMODEL				*/
/*  										*/
/********************************************/






/* notes on n-step fcst
rain_tbl.sequence_opt
{'input_length': 'xlen', 'target_length': 'ylen', 'token_size': 1}


looking further at the docs....

model.score only does a 1-step forecast

(just like the emiting of tokens from llms)


they have implemented model.forecast() as a function

but it is in python. you need to implement it in either SAS or CASL for multi-step fcst

 */