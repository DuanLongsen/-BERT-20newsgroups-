import re
import string
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # ✨新增：用于划分验证集

def preprocess_text(text):
    """简单的文本预处理（主要供作业2的GRU模型使用）"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # 移除HTML标签
    text = text.translate(str.maketrans('', '', string.punctuation))  # 移除标点符号
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = ' '.join(text.split())  # 移除多余空格
    return text

def build_vocab(texts):
    """构建词汇表（供GRU等模型使用，BERT自带Tokenizer无需此步）"""
    word_freq = Counter()
    for text in texts:
        words = text.split()
        word_freq.update(words)

    # 创建词汇表，从1开始（0保留给padding）
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= 2:  # 只保留出现至少2次的词
            word_to_idx[word] = len(word_to_idx)
    return word_to_idx

def get_bert_data():
    """
    加载并拆分20newsgroups数据
    返回: 训练集、验证集、测试集的 raw_text 和 label (专供BERT使用)
    """
    categories = ['alt.atheism', 'soc.religion.christian']   # 无神论vs基督教

    # 1. 下载/加载原始数据
    print("📥 正在加载 20newsgroups 数据...")
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    # 2. 标签编码 (将字符类别转为 0 和 1)
    label_encoder = LabelEncoder()
    y_train_full = label_encoder.fit_transform(newsgroups_train.target)
    y_test = label_encoder.transform(newsgroups_test.target)

    # 3. 划分验证集 (从训练集中按 8:2 比例划分)
    # random_state=42 保证每次运行划分结果一致；stratify 保证正负样本比例均衡
    X_train, X_val, y_train, y_val = train_test_split(
        newsgroups_train.data,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    X_test = newsgroups_test.data

    print("✅ 数据集拆分完成！")
    print(f"📊 训练集样本数: {len(X_train)}")
    print(f"📊 验证集样本数: {len(X_val)}")
    print(f"📊 测试集样本数: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # 测试一下我们的数据脚本运行情况
    X_train, X_val, X_test, y_train, y_val, y_test = get_bert_data()

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # 修复 ImportError: 必须使用原生 PyTorch 的 AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')  # 忽略一些烦人的底层库警告


# ==========================================
# 第一部分：数据加载与预处理
# ==========================================
def get_bert_data():
    """加载、标签编码并拆分 20newsgroups 数据集"""
    categories = ['alt.atheism', 'soc.religion.christian']

    print("📥 [1/4] 正在从网络加载 20newsgroups 数据，请稍候...")
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                          remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    label_encoder = LabelEncoder()
    y_train_full = label_encoder.fit_transform(newsgroups_train.target)
    y_test = label_encoder.transform(newsgroups_test.target)

    # 按照 8:2 划分训练集和验证集，stratify 保证正负样本比例均衡
    print("✂️ [2/4] 正在拆分训练集与验证集 (8:2)...")
    X_train, X_val, y_train, y_val = train_test_split(
        newsgroups_train.data,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    X_test = newsgroups_test.data
    return X_train, X_val, X_test, y_train, y_val, y_test


class NewsDataset(Dataset):
    """自定义 Dataset，将纯文本转化为 BERT 能看懂的 Token IDs"""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==========================================
# 第二部分：BERT 模型构建、训练与评估
# ==========================================
def train_and_evaluate_bert(X_train, X_val, X_test, y_train, y_val, y_test):
    print("🤖 [3/4] 正在下载/加载 BERT-base 预训练模型和 Tokenizer...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    val_dataset = NewsDataset(X_val, y_val, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 自动识别是否有 GPU 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 优化器与学习率调度器 (BERT 微调黄金参数)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3  # 大模型微调只需要 3 轮！
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1),
                                                num_training_steps=total_steps)

    print(f"🚀 [4/4] 一切就绪！开始训练，当前使用设备: {device}")
    print("-" * 50)

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        train_preds, train_labels_list = [], []

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            # ✨ 核心优化：梯度裁剪，把 Loss 稳稳压住！
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(outputs.logits, dim=1).flatten()
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

            # 打印小批次进度（每 20 个 batch 报一次）
            if (step + 1) % 20 == 0:
                print(f"   [Epoch {epoch + 1}] Step {step + 1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels_list, train_preds)

        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1).flatten()
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels_list, val_preds)

        print(
            f"🟢 Epoch {epoch + 1} 总结 | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        print("-" * 50)

    # --- 测试集最终评估 ---
    print("\n🎯 训练完成！正在测试集上进行最终评估...")
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).flatten()

            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    print("\n📊 BERT 模型在测试集上的最终表现：")
    print(classification_report(test_labels_list, test_preds, target_names=['alt.atheism', 'soc.religion.christian'],
                                digits=4))


# ==========================================
# 程序执行入口
# ==========================================
if __name__ == "__main__":
    # 1. 获取拆分好的数据
    X_train, X_val, X_test, y_train, y_val, y_test = get_bert_data()

    # 2. 喂给 BERT 训练
    train_and_evaluate_bert(X_train, X_val, X_test, y_train, y_val, y_test)
