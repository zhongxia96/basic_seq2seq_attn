# basic_seq2seq_attn
the basic seq2seq with attention, it's a baseline of my experiment of neural text summarization. it was written based dataset CNN/Dailymail.

where to find the dataset:



in train stage, run commmand:
python run_summarization.py --mode=train --data_path=the_data_path/chunked/train_* --vocab_path=the_data_path/vocab --log_root=the_log_path --exp_name=the_exp_path

in eval stage, run command:
python run_summarization.py --mode=eval --data_path=the_data_path/chunked/val_* --vocab_path=the_data_path/vocab --log_root=the_log_path --exp_name=the_exp_path

in decode stage, run command:
python run_summarization.py --mode=decode --data_path=the_data_path/chunked/test_* --vocab_path=the_data_path/vocab --log_root=the_log_path --exp_name=the_exp_path
if you want to get the final result and calculate the rouge, run this command:
python run_summarization.py --mode=decode --data_path=the_data_path/chunked/test_* --vocab_path=the_data_path/vocab --log_root=the_log_path --exp_name=the_exp_path --single_pass=True



