from data_providers.ud_pos.pos import DataProvider

pos = DataProvider(lang='russian')
train_pos_tags = pos.train_pos_tags
dev_pos_tags = pos.dev_pos_tags
