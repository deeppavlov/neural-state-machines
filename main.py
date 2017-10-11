from data_providers.ud_pos.pos import DataProvider

pos = DataProvider(lang='russian')
print(pos.train_path)
print(pos.dev_path)
