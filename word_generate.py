import torch
import jieba
import re
from word_poetry_model import WordLevelLSTM


class WordLevelPoetryGenerator:
    """基于词汇的古诗生成器"""

    def __init__(self, model_path='models/word_model_epoch_80.pth'):
        # 加载模型和词汇
        checkpoint = torch.load(model_path, map_location='cpu')

        self.word2idx = checkpoint['vocab']
        self.idx2word = checkpoint['idx2word']
        vocab_size = len(self.word2idx)

        # 创建模型
        self.model = WordLevelLSTM(vocab_size, embed_dim=64, hidden_dim=128)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"模型加载完成，词汇表大小: {vocab_size}")

    def segment_input(self, text):
        """对输入进行分词"""
        words = list(jieba.cut(text, cut_all=False))
        return words

    def generate_poem(self, start_text="春风", style='七言', max_words=16, temperature=0.7):
        """生成古诗

        Args:
            start_text: 起始文本
            style: '五言' 或 '七言'
            max_words: 最大词数
            temperature: 温度参数
        """
        # 分词
        start_words = self.segment_input(start_text)

        # 转换为索引
        input_seq = [self.word2idx.get('<START>', 0)]
        for word in start_words:
            if word in self.word2idx:
                input_seq.append(self.word2idx[word])
            else:
                input_seq.append(self.word2idx.get('<UNK>', 1))

        # 生成
        generated_words = start_words.copy()
        input_tensor = torch.tensor([input_seq[-3:]]).long()  # 使用最后3个词

        hidden = None

        with torch.no_grad():
            for i in range(max_words - len(start_words)):
                # 获取下一个词
                next_idx, hidden = self.model.generate_next_word(
                    input_tensor, hidden, temperature
                )

                # 转换为词
                next_word = self.idx2word.get(next_idx, '<UNK>')

                # 检查是否结束
                if next_word == '<END>' or next_word == '<PAD>':
                    break

                # 添加新词
                generated_words.append(next_word)

                # 更新输入
                input_seq.append(next_idx)
                input_tensor = torch.tensor([input_seq[-3:]]).long()

        # 格式化为诗
        poem_text = self.format_poem(generated_words, style)
        return poem_text

    def format_poem(self, words, style='七言'):
        """将词列表格式化为古诗

        策略：根据五言或七言的要求，智能组合词汇
        """
        if style == '五言':
            target_chars = 5
            total_lines = 4
        else:  # 七言
            target_chars = 7
            total_lines = 8

        poem_lines = []
        current_line = ""
        current_chars = 0

        for word in words:
            word_chars = len(word)

            # 如果添加这个词不会超过目标长度
            if current_chars + word_chars <= target_chars:
                current_line += word
                current_chars += word_chars
            else:
                # 补全当前行
                if current_line and current_chars < target_chars:
                    # 可以添加填充词
                    fill_word = self.find_fill_word(target_chars - current_chars)
                    if fill_word:
                        current_line += fill_word

                # 保存当前行
                if current_line:
                    poem_lines.append(current_line)

                # 开始新行
                current_line = word
                current_chars = word_chars

            # 如果达到目标行数，停止
            if len(poem_lines) >= total_lines:
                break

        # 处理最后一行
        if current_line and len(poem_lines) < total_lines:
            if current_chars < target_chars:
                fill_word = self.find_fill_word(target_chars - current_chars)
                if fill_word:
                    current_line += fill_word
            poem_lines.append(current_line)

        # 添加标点
        formatted_lines = []
        for i, line in enumerate(poem_lines):
            if i % 2 == 0:  # 奇数行
                if i == len(poem_lines) - 1:  # 最后一行
                    formatted_lines.append(line + "。")
                else:
                    formatted_lines.append(line + "，")
            else:  # 偶数行
                formatted_lines.append(line + "。")

        return "\n".join(formatted_lines)

    def find_fill_word(self, needed_chars):
        """寻找合适的填充词"""
        # 简单的填充词查找逻辑
        fill_words = {
            1: ['之', '而', '以', '于', '乎'],
            2: ['明月', '清风', '流水', '高山', '白云'],
            3: ['天地间', '人世间', '古今情', '山水画']
        }

        if needed_chars in fill_words and needed_chars > 0:
            return fill_words[needed_chars][0]
        return None

    def generate_with_constraints(self, start_text, keywords=[], style='七言'):
        """带关键词约束的生成"""
        # 在生成过程中偏好包含关键词
        # 这里可以实现更复杂的约束生成逻辑
        poem = self.generate_poem(start_text, style)

        # 可以在这里添加后处理，确保关键词出现
        return poem


# 主程序
if __name__ == "__main__":
    # 初始化生成器
    generator = WordLevelPoetryGenerator()

    # 生成示例
    print("=" * 50)
    print("基于词汇的古诗生成示例")
    print("=" * 50)

    # 测试不同起始词
    test_cases = [
        ("春风", "七言"),
        ("明月", "五言"),
        ("山水", "七言"),
        ("秋雨", "五言"),
    ]

    for start_word, style in test_cases:
        print(f"\n起始词: '{start_word}' - {style}诗")
        print("-" * 30)

        poem = generator.generate_poem(start_word, style=style, temperature=0.7)
        print(poem)
        print()

    # 交互式生成
    print("\n交互式生成（输入'退出'结束）")
    while True:
        try:
            start = input("\n请输入起始词或短语: ").strip()
            if start == '退出':
                break

            style = input("请选择诗体 (1-五言, 2-七言): ").strip()
            style = '五言' if style == '1' else '七言'

            poem = generator.generate_poem(start, style=style, temperature=0.8)
            print(f"\n生成的{style}诗:")
            print(poem)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"生成失败: {e}")
