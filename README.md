This repo provides an implementation of the paper *Interpreting Twitter User Geolocation*.

## Dataset

You can download GeoText and TwitterUS from [amazondrive](https://www.amazon.com/clouddrive/share/kfl0TTPDkXuFqTZ17WJSnhXT0q6fGkTlOTOLZ9VVPNu/folder/jRda2ADlTYy9XhWB9RUNng?_encoding=UTF8&*Version*=1&*entries*=0&mgh=1) and put them into "./dataset_cmu" and "./dataset_na" folder separately.

The following content will take GeoText dataset as an example of usage.

## Requirements

The code was tested with `Python 3.7`, `tensorflow-gpu 1.14.0` on windows 10. Install the dependencies via Anaconda:

```python
conda install tensorflow-gpu==1.14.0

pip install -r requirements.txt
```

## Usage

This repo includes three main parts:

1. Data processing (script folder: "./data_process", data folder: "./dataset_cmu")
2. Model construction with influence functions (script files: "./main_SGC_inf.py", "./main_N2V_inf.py")
3. Plot analysing  (script folder: "./plot_functions", data folder: "./plot_data")

### Data Processing

```python
cd data_process
python preprocess.py
python handle_n2v.py --input ../dataset_cmu/edge/edge_pair.ungraph  --output ../dataset_cmu/edge/out_of_order.emd  --dimensions 128
```

### Model with IF

```python
python main_SGC_inf.py
python get_influ_matrix.py --ResFolder "./Res_inf_SGC" --SaveFile "./plot_data/sgc_all_inf.txt"
python main_N2V_inf.py
python get_influ_matrix.py --ResFolder "./Res_inf_N2V" --SaveFile "./plot_data/n2v_all_inf.txt"
```

### Plot analysing

```python
cd plot_functions
python plot_degree.py
python plot_comparison.py
python plot_geo_dist.py
python plot_cluster.py
```

## Cite

If you find this work useful for your research, please consider citing us:

>```
>@inproceedings{Ting2020Interpreting,
>	author = {Ting Zhong, Tianliang Wang, Fan Zhou, Goce Trajcevski, Kunpeng Zhang and Yi Yang}, 
>	title = {Interpreting Twitter User Geolocation}, 
>  	booktitle = {the 58th Annual Meeting of the Association for Computational Linguistics (ACL)}, 
>  	year = {2020}, 
>}
>```
