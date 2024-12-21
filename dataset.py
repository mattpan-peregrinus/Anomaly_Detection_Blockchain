import pandas as pd 
import zipfile

zip_path = "/Users/matthewpan/Desktop/spam_token_prediction.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open('spam_token_prediction/transactions.parquet') as f:
        df = pd.read_parquet(f)
        
print (df)
        
 
 

# Files in the zip file 
'''
spam_token_prediction/
spam_token_prediction/training_tokens.parquet
__MACOSX/spam_token_prediction/._training_tokens.parquet
spam_token_prediction/nft_transfers.parquet
__MACOSX/spam_token_prediction/._nft_transfers.parquet
spam_token_prediction/transactions.parquet
__MACOSX/spam_token_prediction/._transactions.parquet
spam_token_prediction/token_transfers.parquet
__MACOSX/spam_token_prediction/._token_transfers.parquet
spam_token_prediction/test_tokens.parquet
__MACOSX/spam_token_prediction/._test_tokens.parquet
spam_token_prediction/dex_swaps.parquet
__MACOSX/spam_token_prediction/._dex_swaps.parquet
'''