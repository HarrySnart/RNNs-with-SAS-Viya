
/********************************************/
/*  										*/
/* 		Multi-Step Forecast with RNN		*/
/*  										*/
/********************************************/

/* the RNN models only produce (emit) a 1-step forecast 

to get a multi-step forecast you need to call the forecast model multiple times inserting the latest forecast value into the series

 */

/* simulate a time series with seasonality and noise */
proc python;
submit;
import numpy as np
import pandas as pd
np.random.seed(2)

T = 20
L = 2000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / T).astype('float64')

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

/* macro for multi-step forecast */

/* logic

- inputs: model table, series variable, output table, copyvars,  name of pred var, number of steps?

logic
- score input table to create initial _DL_Pred_ variable
- create loop for N-1 steps using VARS=_DL_Pred_
- create output table with join 
 */


%macro forecast(model,weights,caslib,input,series,output,copyvar,name,nsteps);

/* model inference for first set of results*/
proc cas;
session casauto;
   deepLearn.dlScore /                                                    /*2*/
      casOut={name="test_scored", replace=TRUE} copyvars={&copyvar} 
      initWeights={name=&weights}
      modelTable={name=&model}
      table={name=&input,vars={&series}} ;
run;

proc cas;
session casauto;
table.fetch / table={name='test_scored'};run;
quit;

%do i=2 %to &nsteps;
/* score predictions */
proc cas;
session casauto;
   deepLearn.dlScore /                                                    /*2*/
      casOut={name="test_scored", replace=TRUE} copyvars={&copyvar} 
      initWeights={name=&weights}
      modelTable={name=&model}
      table={name="test_scored",vars={"_DL_Pred_"}} ;
run;

/* fetch predictions */
proc cas;
session casauto;
table.fetch / table={name='test_scored'};run;
quit;

%end; 

%put %dequote(&caslib.).%dequote(&input.);

data %dequote(&caslib.).%dequote(&output.);
merge casuser.test_scored %dequote(&caslib.).%dequote(&input.);
by date;
rename _DL_Pred_ = &name.;
run;

proc sgplot data=%dequote(&caslib.).%dequote(&output.);
series x=%dequote(&copyvar.) y=%dequote(&series.) / ;
series x=%dequote(&copyvar.) y=%dequote(&name.);
title 'Actual v Forecast';
*where year(date) ge 1999;
run;

%mend;

%forecast(model="test_rnn",weights="test_rnn_weights",caslib="casuser",input="sin_tst",series="y_tx",output="viz_preds",copyvar="date",name="Preds",nsteps=5);


/* the reason this is not working is because i need to re-lag the input varialbes. */

