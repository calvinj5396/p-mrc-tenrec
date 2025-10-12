import pandas as pd
import numpy as np
import os
from pathlib import Path

class TenrecDataAnalyzer:
    """Tenrec数据集分析工具"""
    
    def __init__(self, data_dir='./data/Tenrec'):
        self.data_dir = data_dir
        self.ctr_path = os.path.join(data_dir, 'ctr_data_1M.csv')
        self.qk_video_path = os.path.join(data_dir, 'QK-video.csv')
        
    def check_file_exists(self, filepath):
        """检查文件是否存在"""
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filepath}")
            return False
        print(f"✅ 文件存在: {filepath}")
        file_size = os.path.getsize(filepath) / (1024**3)  # GB
        print(f"   文件大小: {file_size:.2f} GB")
        return True
    
    def analyze_csv_structure(self, filepath, dataset_name, nrows=10000):
        """分析CSV文件结构"""
        print(f"\n{'='*60}")
        print(f"📊 分析数据集: {dataset_name}")
        print(f"{'='*60}")
        
        if not self.check_file_exists(filepath):
            return None
        
        try:
            # 读取前几行快速查看
            print(f"\n🔍 读取前{nrows}行进行分析...")
            df_sample = pd.read_csv(filepath, nrows=nrows)
            
            # 基本信息
            print(f"\n📋 基本信息:")
            print(f"   样本行数（前{nrows}行）: {len(df_sample)}")
            print(f"   特征列数: {len(df_sample.columns)}")
            
            # 列名和数据类型
            print(f"\n📝 所有列名和数据类型:")
            print("-" * 60)
            for i, (col, dtype) in enumerate(zip(df_sample.columns, df_sample.dtypes), 1):
                null_count = df_sample[col].isnull().sum()
                null_pct = (null_count / len(df_sample)) * 100
                print(f"   {i:2d}. {col:25s} | 类型: {str(dtype):10s} | 缺失: {null_count:5d} ({null_pct:.1f}%)")
            
            # 识别特征和标签
            print(f"\n🎯 特征和标签识别:")
            print("-" * 60)
            
            # 可能的标签列（二元标签）
            potential_labels = []
            for col in df_sample.columns:
                unique_vals = df_sample[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    potential_labels.append(col)
            
            if potential_labels:
                print(f"   🏷️  潜在标签列 (二元0/1): {', '.join(potential_labels)}")
            
            # ID类特征
            id_features = [col for col in df_sample.columns if 'id' in col.lower() or 'user' in col.lower() or 'item' in col.lower()]
            if id_features:
                print(f"   🆔 ID类特征: {', '.join(id_features)}")
            
            # 类别特征
            categorical_features = []
            for col in df_sample.columns:
                if col not in potential_labels and df_sample[col].dtype == 'object':
                    categorical_features.append(col)
                elif col not in potential_labels and df_sample[col].dtype in ['int64', 'float64']:
                    unique_count = df_sample[col].nunique()
                    if unique_count < 100 and col not in id_features:
                        categorical_features.append(col)
            
            if categorical_features:
                print(f"   📊 类别特征: {', '.join(categorical_features)}")
            
            # 数值特征
            numeric_features = []
            for col in df_sample.columns:
                if col not in potential_labels and col not in id_features and col not in categorical_features:
                    if df_sample[col].dtype in ['int64', 'float64']:
                        numeric_features.append(col)
            
            if numeric_features:
                print(f"   🔢 数值特征: {', '.join(numeric_features)}")
            
            # 统计信息
            print(f"\n📈 标签分布统计:")
            print("-" * 60)
            for label in potential_labels:
                value_counts = df_sample[label].value_counts()
                print(f"\n   {label}:")
                for val, count in value_counts.items():
                    pct = (count / len(df_sample)) * 100
                    print(f"      {val}: {count:6d} ({pct:5.2f}%)")
            
            # 类别特征的唯一值统计
            print(f"\n📊 类别特征唯一值数量:")
            print("-" * 60)
            for col in categorical_features[:10]:  # 只显示前10个
                unique_count = df_sample[col].nunique()
                print(f"   {col:25s}: {unique_count:8d} 个唯一值")
            
            # 数值特征的统计
            if numeric_features:
                print(f"\n🔢 数值特征统计摘要:")
                print("-" * 60)
                print(df_sample[numeric_features].describe())
            
            # 显示前5行数据样例
            print(f"\n📄 数据样例（前5行）:")
            print("-" * 60)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df_sample.head())
            
            return {
                'columns': list(df_sample.columns),
                'dtypes': df_sample.dtypes.to_dict(),
                'potential_labels': potential_labels,
                'id_features': id_features,
                'categorical_features': categorical_features,
                'numeric_features': numeric_features,
                'sample_data': df_sample
            }
            
        except Exception as e:
            print(f"❌ 分析出错: {str(e)}")
            return None
    
    def compare_datasets(self, ctr_info, qk_info):
        """对比两个数据集"""
        if not ctr_info or not qk_info:
            return
        
        print(f"\n{'='*60}")
        print(f"🔄 数据集对比: ctr_data_1M vs QK-video")
        print(f"{'='*60}")
        
        ctr_cols = set(ctr_info['columns'])
        qk_cols = set(qk_info['columns'])
        
        common_cols = ctr_cols & qk_cols
        ctr_only = ctr_cols - qk_cols
        qk_only = qk_cols - ctr_cols
        
        print(f"\n📊 列对比:")
        print(f"   共同列数: {len(common_cols)}")
        print(f"   仅在ctr_data_1M: {len(ctr_only)}")
        print(f"   仅在QK-video: {len(qk_only)}")
        
        if common_cols:
            print(f"\n   ✅ 共同列: {', '.join(sorted(common_cols))}")
        
        if ctr_only:
            print(f"\n   ➕ ctr_data_1M特有: {', '.join(sorted(ctr_only))}")
        
        if qk_only:
            print(f"\n   ➕ QK-video特有: {', '.join(sorted(qk_only))}")
        
        print(f"\n🏷️  标签对比:")
        print(f"   ctr_data_1M标签: {', '.join(ctr_info['potential_labels'])}")
        print(f"   QK-video标签: {', '.join(qk_info['potential_labels'])}")
    
    def run_full_analysis(self, sample_rows=10000):
        """运行完整分析"""
        print(f"\n{'#'*60}")
        print(f"# Tenrec 数据集分析工具")
        print(f"# 分析样本行数: {sample_rows}")
        print(f"{'#'*60}")
        
        # 分析ctr_data_1M
        ctr_info = self.analyze_csv_structure(
            self.ctr_path, 
            'ctr_data_1M.csv',
            nrows=sample_rows
        )
        
        # 分析QK-video
        qk_info = self.analyze_csv_structure(
            self.qk_video_path,
            'QK-video.csv',
            nrows=sample_rows
        )
        
        # 对比分析
        self.compare_datasets(ctr_info, qk_info)
        
        print(f"\n{'='*60}")
        print(f"✅ 分析完成！")
        print(f"{'='*60}")
        
        return ctr_info, qk_info


# 使用示例
if __name__ == "__main__":
    # 创建分析器
    analyzer = TenrecDataAnalyzer(
        data_dir='./data/Tenrec'
    )
    
    # 运行分析（默认读取前10000行）
    # 如果数据集很大，可以增加sample_rows参数
    ctr_info, qk_info = analyzer.run_full_analysis(sample_rows=100)
    
    # 如果需要分析更多行，可以这样：
    # ctr_info, qk_info = analyzer.run_full_analysis(sample_rows=100000)