import numpy as np
import tensorflow as tf
import pandas as pd
import re

def templatise(message: str) -> str:
    """
    Replace dynamic values with placeholder tokens so the vectorizer
    learns log STRUCTURE, not random IDs/numbers.
    e.g. "user_id=ABC123 latency=342ms" → "user_id=<ID> latency=<LAT>"
    """
    msg = str(message)
    msg = re.sub(r'\b\d{1,3}(\.\d{1,3}){3}\b', '<IP>',  msg)   # IP addresses
    msg = re.sub(r'\b[A-Z0-9]{6,}\b',           '<ID>',  msg)   # hex / block IDs
    msg = re.sub(r'\d+ms',                       '<LAT>', msg)   # latency values
    msg = re.sub(r'\d{3,}',                      '<NUM>', msg)   # long numbers
    return msg

def get_data(max_len=50,batch_size=64,trace_path=None,label_path=None,val_split=0.15):
    traces_df = pd.read_csv(trace_path)
    label_df=pd.read_csv(label_path)
    
    traces_df = traces_df.rename(columns={'Label': 'New_Label'})
    # traces_df=traces_df.drop(columns=['Label'],errors='ignore')
    merged_df=pd.merge(traces_df,label_df,on="BlockId")
    
    merged_df['Features']=(
    merged_df['Features']
    .astype(str)
    .str.replace(',',' ',regex=False)
    .str.replace('[',' ',regex=False)
    .str.replace(']',' ',regex=False)
    .str.strip()
    )
    
    merged_df['Features']=merged_df['Features'].apply(templatise)
    
    # merged_df['Label_Int']=merged_df['Label'].astype(int)
    # merged_df['Label_Int'] = merged_df['Label'].map({'Normal': 0, 'Anomaly': 1})
    
    merged_df['Label_Int'] = (merged_df['Label'].str.strip().str.lower() != 'normal').astype(int)
    assert merged_df['Label_Int'].isna().sum() == 0, "Unexpected label values found"
    
    normal_df=merged_df[merged_df['Label']=='Normal'].reset_index(drop=True)
    all_seqs=merged_df['Features'].tolist()
    normal_seqs=normal_df['Features'].tolist()
    
    vectorized=tf.keras.layers.TextVectorization(
        max_tokens=10000,
        output_sequence_length=max_len,
        standardize='lower_and_strip_punctuation'
    )
    
    vectorized.adapt(all_seqs)
    X_normal=vectorized(normal_seqs)
    
    n=len(X_normal)
    val_size=max(1,int(n*val_split))
    indx=tf.random.shuffle(tf.range(n),seed=42)
    train_indx=indx[val_size:]
    val_indx=indx[:val_size]
    
    X_train=tf.gather(X_normal,train_indx)
    X_val=tf.gather(X_normal,val_indx)
    
    train_ds=(
        tf.data.Dataset.from_tensor_slices((X_train,X_train))
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    val_ds=(
        tf.data.Dataset.from_tensor_slices((X_val,X_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    X_all=vectorized(all_seqs)
    y_all=merged_df['Label_Int'].values
    
    return train_ds,val_ds,X_val,X_all,y_all,vectorized