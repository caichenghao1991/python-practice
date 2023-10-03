'''
数据库 http://archive.ics.uci.edu/ml/
Microsoft Azure Machine Learning studio: 实验左侧所有操作功能，中间流程图连接各个操作，右侧具体当前操作参数
kaggle， 天池  机器学习比赛
ssl 883 error 解决方法:
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context


pytorch：http://121.199.45.168:8020/1/
自然语言处理入门：http://121.199.45.168:8005/
文本预处理：http://121.199.45.168:8003/1/#11
经典序列模型：http://121.199.45.168:8004/1/
RNN及其变体：http://121.199.45.168:8002/1/
Transformer：http://121.199.45.168:8001/1/
迁移学习：http://121.199.45.168:8007/1/

1. 机器学习概述
    1.1 人工智能概述
        a. 人工智能起源
            图灵测试： 区分人还是机器
            达特茅斯会议： 定义人工智能， 人工智能起点
        b. 人工智能三个阶段
            1980 年代是正式形成期
            1990-2010 年是蓬勃发展期
            2012年后是深度学习期
        c. 人工智能、机器学习和深度学习
            机器学习是人工智能的一个实现途径
            深度学习是机器学习的一个方法（人工神经网络）发展而来
        d. 人工智能主要分支
            计算机视觉
            自然语言处理：语音识别，语义识别（文本挖掘/分类，机器翻译）
            机器人
        e. *** 人工智能三要素
            数据
            算法
            计算力
        f. ** CPU 擅长读写文件(IO密集型)， GPU 计算密集型（数据计算，易于并行的程序）

    1.2 机器学习工作流程
        a. *** 定义
            机器学习是从数据中自动分析获得模型，并利用模型对未知数据进行预测
        b. **** 工作流程
            获取数据
            数据处理
            特征工程
            机器学习（模型训练）
            模型评估 （效果不好，则重新执行以上步骤）
            应用
        c. ***** 获取到的数据集介绍
            i. 专用名词
                一行数据称为样本 (instance)
                一列数据称为特征 (feature)
                目标值 (需要预测的标签值，连续/离散。由数据库获得，或专家标记)
                特征值（已知的列数据）
            ii.数据类型构成
                特征值 + 目标值
                只有特征值
            iii. 数据分割
                训练集 -- 构建模型
                （验证集 -- 参数和模型调整）
                测试集 -- （最终）模型评估
        d. 数据基本处理
            对数据缺少值，去异常值等处理
            pandas 数据清洗，处理
        e. 特征工程
            i. 定义：使用专业背景知识和技巧处理数据，使得特征能在机器学习算法上发挥更好的作用的过程
            数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。
            ii. 包含内容：
                特征提取（特征值化）： 将数据转（字典，文本，离散类别字符串，图像）化为可用于机器学习的数字特征
                特征预处理： 通过转换函数将特征数据转换成更加适合算法模型的特征数据过程
                特征降维： 降低随机变量（特征）个数，简化模型的过程
            sklearn 特征工程
        f. 机器学习
            选择合适的算法对模型进行训练
        g. 模型评估
            对训练好的模型进行评估

    1.3 完整机器学习项目流程
        a. 抽象成数学问题
            i. 明确问题
            ii. 通过可以获得的数据抽象出是分类/回归/聚类等的问题
        b. 获取数据
            i. 数据要有代表性，防止过拟合
            ii. 分类问题数据不能偏移过于严重
            iii. 对于数据量级的评估，估算内存消耗程度，是否需要降维，改进算法或分布式
        c. 特征处理与特征选择
            筛选出显著特征可以提高算法的效果和性能
            常用操作
                归一化
                离散化
                因子化
                缺失值处理
                去除共线性
                特征有效性分析 （相关系数，卡方检验，平均互信息，条件箱，后验概率，逻辑回归权重）
        d. 训练模型与调优
            调整算法超参数使得结果更佳（需要对算法原理的深入理解）
        e. 模型诊断
            通过模型诊断确定模型调优的方向与思路

    1.4 *** 机器学习算法分类
        a. 监督学习 （预测结果）
            输入数据（独立同分布）由特征值和目标值组成。目标值是连续值（回归），是离散值（分类）
            分类： k-近邻，贝叶斯分类，决策树与随机森林，逻辑回归，神经网络
            回归： 线性回归，岭回归
        b.无监督学习 （发现潜在结构）
            输入数据由特征值组成。输入数据无标签值，需要根据样本间的相似性对样本进行分类（聚类），使类内差距最小，类间差距最大
            聚类k-means， 降维
        c. 半监督学习
            输入数据有特征值，但是部分数据有目标值，部分没有。
        d. 强化学习
            动态过程，上一步数据的输出是下一步数据的输入。学习基于状态的决策方案。代理通过行动来操作环境，从一个转态转变到另一个状态，奖励函数根
            据是否完成（子）任务来判断行为好坏（可能有延迟），并分配奖励。代理通过开发(exploitation)和探索（exploration）之间的权衡，选择最
            大的回报。
            四要素：代理（agent），行为（action），环境（environment），奖励（reward）
            马尔科夫决策，动态规划

    1.5 模型评估
        a. 分类模型评估
            准确率：预测正确的数占样本总数的比例
            精确率：正确预测为正占全部预测为正的比例
            召回率：正确预测为正占全部正样本的比例
            F1-score：主要评估模型的稳健性
            AUC：主要用于评估样本不均衡的情况
        b. 回归模型评估
            实际值 a, 预测值 p, 样本数 n
            均方根误差（root mean squared error, RMSE），仅能比较误差是相同单位的模型。 sqrt(sum((p-a)^2) / n)
            相对平方误差（relative squared error, RSE），能比较误差是不同单位的模型。 sum((p-a)^2) / sum((mean(a)-a)^2)
            平均绝对误差（mean absolute error, MAE），仅能比较误差是相同单位的模型。 sum(abs(p-a)) / n
            相对绝对误差（relative absolute error），能比较误差是不同单位的模型。 sum(abs(p-a)) / sum(abs(mean(a)-a))
            决定系数（R^2） 值越接近 1，存在越强的线性关系。 R^2 = 1 - sum((p-a)^2) / sum((mean(a)-a)^2)
        c. 拟合
            模型评估用于评价训练好的模型的表现效果，分为欠拟合（学习到的特征太少，区分太粗糙）和过拟合（学习了太多针对训练集的特征）

    1.6 深度学习简介
        卷积神经网络增加层数来增加更多抽象的概念（物体层，器官层，分子层...），通过增加节点数来增加物质的种类

2 机器学习基础环境安装与使用
    2.1 库的安装
        python<version> -m venv <virtual-environment-name>
        source env/bin/activate
        deactivate
        pip install matplotlib, numpy, pandas, tables, jupyter
    2.2 ** Jupyter Notebook (Julia Python R)
        a. 简介
            开源的科学计算平台，类别ipython，可以运行代码，做笔记，画图表。文件后缀 .ipynb， 具有画图，数据展示方面的优势，
        b. 安装运行
            pip install jupyter
            pip install jupyter_contrib_extensions  安装插件（可选）
                jupyter contrib nbextension install --user --skip-running-check
                额外勾选 table of contents, hinterland
            jupyter notebook 运行
        cell：一对 In Out 会话，被视做一个代码单元
        help(方法名) 显示开发文档
        %time  a=a*2  返回方法执行使用时间
        c. 模式快捷键
            两种模式： 编辑模式（直接点击编写代码），命令模式（Esc 进入，快捷键操作）
            常用快捷键：
                通用
                    Shift + Enter: 执行本单元代码并跳转到下一单元
                    Ctrl + Enter: 执行本单元代码
                命令模式
                    Y: 切换到 Code 模式
                    M： 切换到 Markdown 模式
                    A： 在当前 cell 的上面添加 cell
                    B: 在当前 cell 的下面添加 cell
                    双击 D：删除当前cell
                编辑模式
                    Ctrl + c : 复制
                    Ctrl + v : 粘贴
                    Ctrl + z : 回退
                    Ctrl + y : 重做
                    Tab: 代码补全
                    Ctrl + / : 注释 / 取消注释
        d. Markdown 语法：
            # ： 一级标题
            ## ： 二级标题
            - ： 列表缩进
                -: 二级列表缩进 （tab)

    2.3 Matplotlib (Matrix plot library)
        a. 用途
            绘制 2D (3D) 图表，实现数据可视化，更易于数据分析
        b. *** 绘图流程
            创建画布 plt.figure(figsize=(10, 8), dpi=80)
            绘制图像 plt.plot(x, y)
            图片保存 plt.savefig('test.png')    需要在 show 之前
            显示图像 plt.show()  会清空 plt 对象
        c. 三层结构
            容器层
                Canvas: 底层画板
                Figure： 画布角色，可包含多个 Axes
                Axes(坐标系)： 绘图区的角色（画多个图），可包含多个 axis（坐标轴）
            辅助显示层
                Axes 外观 facecolor, 边框线（spines），坐标轴（axis），坐标轴名称（axis label），坐标轴刻度（tick），坐标轴刻度标签
                （tick label），网格线（grid），图例（legend），标题（title）
            图像层
                绘制不同类型的图像 plot（折线图，绘制函数，表示数据变化）, scatter（数据关联、分布趋势）, bar(统计，对比数据之间差别),
                histogram（展示连续数据的分布状况）, pie（分类数据的占比情况）
        如果中文乱码，需要下载 SimHei 字体，安装后删除 .matplotlib, 修改 matplotlibrc

        **** 实例
            x, y1, y2 = range(60), [random.uniform(20,25) for _ in x], [random.uniform(10,15) for _ in x]
            plt.figure(figsize=(10, 15), dpi=100)
            plt.plot(x, y1, label='上海')
            plt.plot(x, y2, color='r', linestyle='--', label='北京')
            plt.scatter(x, y)
            plt.bar(x, width=0.5, align='center')
            plt.bar(x, y)    # x，y 必须为数字
            plt.hist(x, bins=None, range=None)    range 元组，表示 bins 的下界和上界
            plt.pie(x, labels=, colors=, autopct='%.2f%%')    # autopct 占比显示指定 .2f
            # 辅助显示层需要在图像层之后
            x_ticks, y_ticks = ["11点{}分".format(i) for i in x], range(40)
            plt.xticks(x[::5], x_ticks[::5])  # 第一个参数必须为数字
            plt.yticks(y_ticks[::5])
            plt.grid(True, linestyle="--", alpha=0.5)   # true 为添加网格， alpha 透明度
            plt.xlabel("时间", fontsize=16)
            plt.ylabel("温度", fontsize=16)
            plt.title("11点到12点气温变化", fontsize=20)
            plt.legend(loc='best')  # 显示图例 plot 必须有 label
            plt.show()

            多个坐标系显示图像
            plt.figure(figsize=(10, 15), dpi=100)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
            axes[0].plot(x, y1, label='上海')
            axes[1].plot(x, y2, color='r', linestyle='--', label='北京')
            axes[0].set_xticks(x[::5])
            axes[0].set_xticklabels(x_ticks[::5])
            axes[0].grid()  # .set_xlabel(), .set_title()

    2.4 seaborn
        基于 Matplotlib 进行了更高级 API 的封装，出图更美观
        pip install seaborn
        import seaborn as sns
        散点图
            sns.lmplot(data=df, x='col1', y='col2', hue='col3',fit_reg=False)
                hue 指定目标值对应列名，根据目标值给点涂成不同颜色， fit_reg 是否进行线性拟合， 代替 plt.scatter，其余 plt.xlabel等不变

    2.5 Numpy (numerical Python)
        a. 定义
            numpy 是一个开源的用于快速处理多维数组的库
            存储对象是 ndarry， 使用 np.array([], dtype=np.int16) 多维数组创建
            ndarry 数据类型： bool, int, float, string_, unit (正整数)，complex，object_(python对象)，unicode_

        b. 优势 （速度快，有很多数组运算的 api）
            python 原生数组为了满足存不同类型数据，需通过数组存储的内存地址一个个查询对应的数据，而 numpy 直接将同一数据类型的数组
                （不同类型会从 int -> float -> string）存在连续的地址中
            numpy ndarry 支持并行化（向量化）运算
            numpy 和 python 同为 C 语言编写， 但是没有 Python 的全局解释器锁（GIL）限制

        c. ** N 维数组 ndarry
            i. 属性
                ndarry.shape 数组形状的元组
                ndarry.ndim  数组的维度
                ndarry.size  数组中的元素数量
                ndarray.itemsize  一个数组元素的长度（字节）
                ndarry.dtype 数据类型
            ii. ** 生成数组
                生成 0，1，同值数组
                    a = np.ones([3,5])    zeros()
                    np.ones_like(a)       zeros_like()
                    np.full([3,5],0)
                从现有数组中生成
                    b = np.array(a), np.copy(a)  # 深拷贝, 改 a 值不变
                    c = np.asarray(a)   # 浅拷贝，改 a 值会变, 改数组形状不变
                生成固定范围数组
                    np.arange(10)      默认start=0, step=1, 数组不包含 stop
                    np.linspace(1,10,5)  start, stop, num, endpoint 默认 true（包含 stop）, 等间隔，数组大小 num 为 5
                    np.logspace(0,2,3)   10^0, 到 10^2， 取3个数， 指数等间隔
                生成随机数组
                    均匀分布
                        np.random.rand(3,5)   生成形状 3*5  [0,1) 均匀数组
                        np.random.uniform(0,10,(3,5))   生成形状 3*5 [0,10) 均匀数组
                        np.random.randint(0,10,(3,5))   生成形状 3*5 [0,10) 均匀整数数组
                    *** 正态分布
                    标准差： sigma = sqrt(sum((x-mean(x))^2)/n)
                    正态分布： e^(-(x-mean(x))^2/(2*sigma^2)) / (sigma*sqrt(2*pi)))
                    np.random.randn(3,5)  生成3*5， 均值0,标准差1 正态分布数组
                    np.random.standard_normal((3,5))  生成3*5， 均值0,标准差1 正态分布数组, 和 randn 仅输入参数不同
                    np.random.normal(0,10,(3,5))  生成3*5，均值0,标准差10 正态分布数组

            iii. **** 数组索引切片
                索引从外层到内层
                a = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
                a[0,1,2]  # 6
                a[0,0:1]  # [1,2,3]
                a[0,0,0:2]  # [1, 2]

            iv. ** 形状修改
                a = np.array([[1,2,3],[4,5,6]])
                a2 = a.reshape([3,2])  # 原数组元素从首到尾排成 1 行，再依次排列到新数组大小。
                a2 = a.reshape([-1,2])  # -1 表示未知大小，可自动计算该尺寸，不能得到整数尺寸时会报错
                a.resize([3,2])    # 对 a 原地修改，其他同 reshape
                a3, a4 = a.T, a.transpose()  # 转置 (行列互换)

            v. * 类型修改
                a.astype(np.int32)
                a.tostring()   a.tobytes()   转化为 byte 对象

            vi. * 数组的去重
                a = np.unique(a)

            vii. ** 数组的运算
                逻辑运算
                    a = np.array([[1,2,3],[4,5,6]])
                    a > 3          [[False, False, False], [ True,  True,  True]]
                    a[a>3] = 2     [[1,2,3],[2,2,2]]   大于 3 的值重新赋值为 2
                通用判断函数
                    np.any(a<2)  True  所有满足要求，才返回 True
                    np.all(a>0)  True  只要有一个满足要求，就返回 True
                三元运算符
                    np.where(a>3,a,0)  大于 3 的值赋值为 a，其他为 0   满足要求则使用第一个值赋值，否则使用第二个值赋值
                    np.where(np.logical_or(a>3,a<2),a,0)  大于 3 或者小于 2 的值赋值为 a，其他为 0
                        np.logical_and    np.logical_not    np.logical_xor
                统计运算
                    np.min(a, axis=1)     max, median, mean, std, var
                    np.argmax(a, axis=1)   返回每个第 1 维度中最大值所在下标
                    a.max() 为数组全部值的最大值

        d. * 矩阵运算
            向量： 一维数组    矩阵：二维数组
            矩阵加法：一对一同索引元素相加
            矩阵与标量乘法： 矩阵每个元素都乘以这个标量
                numpy 数组与常数: a + b   a + 1    a / 2
                    数组元素与数组元素相乘使用广播机制（两数组 shape 靠右对齐， 检查每组维度，如果对于所有的维度组（两个值相等或 其中一个值为
                    1或空则）则可以进行广播。  如 A 15 *3 *5,  B: 15*1*5 可以广播相乘  a * b 或 np.multiply(a,b)
            *** 矩阵与向量（矩阵）乘法： 结果矩阵第 i 行第 j 列的值为第一个矩阵（向量） 第 i 行 与二个矩阵（向量）第 j 列 的点积。
                c_ij = sum_k(a_ik * b_kj)
                [M行, N列] * [N行, L列] = [M行, L列]  满足结合律，不满足交换律
                numpy 点乘 np.matmul(a,b) 或 矩阵相乘 np.dot(a,b)   （matmul不能矩阵与数字相乘，dot可以，其他相同）
            A * A^-1 = I  对于方阵，矩阵乘以逆矩阵等于单位矩阵（主对角线为 1， 其余为0）
                通过待定系数法或初等变化求逆（矩阵右侧添加单位矩阵，通过矩阵变化将左侧元矩阵变化为单位矩阵，此时右侧的原单位矩阵为原矩阵的逆）
            矩阵的转置（行列互换）

    2.6 pandas (panel data analysis)
        a. 定义
            开源的数据挖掘库，用于数据探索。以 numpy 为基础（高性能），基于 matplotlib（画图），有独特的数据结构
        b. 优势
            便捷的数据处理能力，读取文件，封装了 matplotlib,numpy
        c.Series
            带索引的一维数组
            s = pd.Series([1,2,3], index=list('ABC'), dtype=np.int32)
            s.unique()
            自定义运算
                s.apply(lambda x:2*x)      Series can't use max() for lambda
                s.transform(lambda x:x.max()-x)
                s.map(lambda x:2*x)
        d. DataFrame (Series 的容器)
            具有行索引 index=0, 列索引 index=1, 的二维数组。每列每行都是 Series, 单个值是numpy 数据（numpy.int64,...）
            i. 创建
                df = pd.DataFrame(data, index=[] columns=pd.date_range(start='2017-01-01', periods=data.shape[0], freq='B'))
                    data ndarry, index 行索引， columns 列索引
            ii. 属性
                data.shape
                data.index   data.index.names
                data.columns
                data.values
                data.T  (行列互换)
                data.head(5)  前（默认）5行
                data.tail()

            iii. 索引
                设置索引
                    df.index = ['stock'+str(i) for i in range(data.shape[0])]    修改行索引
                重设索引（0开始的整数）
                    df.reset_index(drop=False, inplace=True)
                设置新索引（以某列值）
                    df.set_index(keys='month',drop=True, inplace=True)  keys 可以是列表，复合索引（数据变成类似多维数组）
                使用索引
                    df['col_name']     返回一列（Series）
                    df[['col_name']]     返回一列（DataFrame）
                    df.loc['row_new']   返回一行（Series）
                    df['col_name']['row_index']   必须先列索引，再行索引，获取一个 cell 的值
                    df.loc['row_index1':'row_index2','col_name']  必须先行索引，再列索引，获取一个 cell 的值
                    df.iloc[:row_index_num, col_num]   必须先行索引数字下标，再列索引数字下标，获取一个 cell 的值
                    df.ix[:row_index_num, ('col_num1','col_num2')]   混合索引，先行后列，获取一个 cell 的值,不推荐
                    df.columns.get_indexer([col_name1'])   获取 col_index1 对应的数字下标
                赋值
                    使用索引获取单元格，直接赋值
                多级索引
                    df.set_index(["year","month"], inplace=True))
                    df.index.names    返回索引列名
                    df.index.levels   返回每个索引列所有的独特值排序后的列表
            iv. 新建、删除行列
                df = pd.DataFrame([[1,2],[3,4]], columns=['a','b'])
                新建列赋
                    df['col_name'] = 1                 # 一列全赋值为1
                    df.col_name = 1
                    df['col_name'] = ['a','b','c']
                新建行
                    df.loc['row_new'] = pd.Series([[5,6],index=['a','b']])   必须与原dataframe 列名相同
                    df = df._append(pd.DataFrame([[5,6]], columns=df.columns, index=df.shape[0]+1))
                        必须申明行列索引，否则默认起始为 0 的递增数列
                删除行 / 列
                    df = df.drop(row_num)    删除行
                    df.drop(['col_name1', 'col_name2'], axis=1, inplace=True)  删除列

            v. 排序 (支持 Series)
                按内容排序
                    df.sort_values(by='col_name', ascending=True, inplace=True)
                        by=['col_1','col_2']  如果第一列值相同根据第二列排序,
                对索引排序
                    df.sort_index()    默认 ascending=True
            vi. 运算 (支持 Series)
                算术运算
                    df['a'] = df['a'].add(2)    df.sub(2)   df + 1    df - 2
                逻辑运算
                    df['col_name']>0   返回 True/False Series      < > | &
                    df[(df['col_name']>0) & (df['col_name']<5)]  返回 df 中 col_name 大于 0 小于 5 的 DataFrame
                逻辑运算函数 (query 不支持 Series)
                    df.query('col_name>0 & col_name<5')    返回 DataFrame (Series 没有此方法)
                    df['col_name'].isin([1,2])   返回 True/False Series col_name值是否是 1 或 2
                统计运算
                    df.describe()    每一列的 count, mean, std, min, 25%, 50%, 75% max
                    df.sum(axis=0)   0： 列求统计结果  1：行求统计结果
                        mean(), median(), min(), max(), mode(), abs(), prod(), std(), var(), idxmax(), idxmin() (索引值)
                累计统计函数
                    df.cumsum(), df.cumprod()     返回 DataFrame， 每一列中前 1,2,3,...,n 个数的和，积
                    df.cummax(),  df.cummin(),    返回 DataFrame， 每一列中前 1,2,3,...,n 个数的最大值，最小值
                    df.cumsum(axis=0)  df.cumprod(axis=0)
                自定义运算
                    df.apply(lambda x:x.max()-x, axis=0)      Series can't use max() for lambda, func 可以返回一个值, Series 或 DataFrame
                    df.apply(lambda x:x['row_index1']-x['row_index2'])
                    df.transform(lambda x:x.max()-x, axis=0)        transform func 不能返回 DataFrame, 只能操作单个 Series
                    def inspect(x):
                        print(type(x))
                    df.apply(inspect)
        e. Panel (DataFrame 的容器，已经过时，使用 MultiIndex 的 DataFrame 代替)
            Panel 是一个 3 维的 DataFrame， 索引为 3 个 axis， 分别是 items, major_axis, minor_axis
            p=pd.Panel(np.arange(24).reshape(4,3,2), items=list('ABCD'), major_axis=range(4), minor_axis=range(2))
            p['A']   返回 items DataFrame
            p.major_xs(2)   返回 major_axis=2 的 DataFrame
            p.minor_xs(1)   返回 minor_axis=1 的 DataFrame
        e. 画图 （支持 Series）
            import matplotlib.pyplot as plt
            df.plot()    kind='line' 折线图默认， 'bar', 'barh', 'hist', 'pie', 'scatter', 'box', 'kde', 'area'
            plt.show()
        f. 文件读取与存储
            df.to_csv('file_name.csv', sep=',', columns=None, header=True, index=True, mode='w', encoding=None)
            df = pd.read_csv('file_name.csv', sep=',', usecols=['col1','col2'])      usecols 读取指定列
            df = pd.read_hdf('data.h5', key='col1')    速度快，压缩节约空间，支持 hadoop，推荐
            df = pd.read_json('data.json', orient='record', typ='frame',lines=False)      typ： frame 或 series
                orient: json格式, split,records,index,columns(默认),values     lines: 按照每行读取（默认False）
            html, clipboard, excel, parquet, pickle, sql, gbq, dict
        g. **** 缺失值处理 （支持 Series）
            np.NaN    缺失值是 float 类型
            pd.isnull(df)    # 返回 True False DataFrame
            pd.notnull(df)
            np.all(pd.notnull(df))   如果有缺失值，返回 False
            np.any(pd.isnull(df))    # 如果有缺失值，返回 True
            df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
            df['col1'].fillna(value=df['col1'].mean(), method='pad', axis=None, inplace=False, limit=None)
            df.replace(to_replace='?', value=np.nan)   替换缺失值
        h. 数据离散化 （支持 Series）
            减少给定连续属性的个数， 在连续属性的值域上，将值域划分为若干个离散的区间，用不同的符合或整数值代表落在每个子区间中的属性值
            s = pd.qcut(df['col'],q=10)     不能输入2D DataFrame,大致将数据分为 10 组（输入必须 Series， 返回Series,值为原值所在的区间）
            s = pd.cut(df['col'],bins=[-10,10])   不能输入2D DataFrame,指定分组间隔，将数据分为小于-10，-10到10，大于10三组
            s.value_counts()    统计分组后每个区间数据的个数

            哑变量/热独编码（onehot encoding）
                dummies = pd.get_dummies(s,prefix=None)    输入可以是 DataFrame, Series, array, prefix 为列名前缀
        i. 数据合并 （concat 支持 Series, merge 不支持）
            pd.concat([df1,df2],axis=1)  df1, df2 左右并排合并（共享行索引）
            pd.concat([df1,df2],axis=1,join='outer',ignore_index=False)
            pd.merge(df1,df2, how='inner', on=['col1','col2'], left_on=None, right_on=None, left_index=False, right_index=False)
                # on 为公共列，默认inner, df1,df2中公共列值相同的行才保留。left_on 表示左侧列表需要保留的列，left_index 表示以左侧
                DataFrame的索引列为连接键
        j. 交叉表与透视表 （crosstab 支持 Series, pivot_table 不支持）
            df['day'] = pd.to_datetime(data.index).weekday     # 指定时间单位为秒
            pd.DatetimeIndex(pd.to_datetime(data.index,unit='s')).month
            pd.crosstab(df['day'],df['col'])    交叉表，以第一项 day 为索引，显示 col 在原表中出现次数， 寻找之间关系
                pd.crosstab(df['day'],df['col'],values=df['col2'],aggfunc=np.sum()) 显示 col 在原表中 col2 所有值的和
            pd.pivot_table(df,index=['col1','col2'],values=['col3'])  改变 DataFrame 或 Series 并变形原表到以index 为新行索引，
                values为列索引，其余列变为复合内层列索引
        k.分组与聚合 （支持 Series）
            groupby  返回 DataFrameGroupBy 对象，通常配合聚合函数
            df.groupby(by=['col1'])['col2'].mean()  对 DataFrame 按列索引分组，对一列进行聚合操作
            df['col2'].groupby(by=['col1','col2'], as_index=False).mean()    as_index=False 额外添加起始 0 自然数索引
            df.groupby('col1').agg({'col1': ['sum', 'mean'], 'col2': 'min'})    同时返回多组聚合结果
            agg(sum), agg(['sum', 'mean'])

3 机器学习算法
    近似误差：对现有训练集的训练误差，在训练集上的表现。
    估计误差：对现有测试集的测试误差，关注测试集，体现模型对未知数据的表现。
    sklearn 估计器，机器学习的父类，实例化 estimator， 调用 fit(x,y), predict(x), score(x,y) 方法

    3.1 scikit-learn
        文档多，且规范。包含的算法多，实现容易。但是不支持神经网络和深度学习。
        包含分类，回归，聚类，降维，模型选择和预处理。
        d = sklearn.datasets.load_*()     ex. load_iris 获取库中本地小规模数据集
        d = sklearn.datasets.fetch_*(data_home=None, subset='train')    ex.fetch_20newsgroups 获取属于库在网络中较大规模数据集，
            默认下载到 ~/scikit_learn_data 目录下, subset 可选 train(默认), test, all, 中途超时，尝试多次下载
            返回数据 Bunch 类似字典，data(特征数组 二维 numpy ndarray)，target(标签数组 一维 ndarray)，feature_names(特征名 list)，
            target_names(ndarray 目标值所有可能取值), DESCR(字符串数据描述)   使用 d.data 或 d["data"] 获取
        from sklearn.model_selection import train_test_split
        from sklearn.externals import joblib
        x_train, x_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.2, random_state=None)
        trans = StandardScaler()
        x_tra = trans.fit_transform(x_train)
        x_tes = trans.transform(x_test)
            将数据集分为训练集和测试集, 测试集占 20%， 传入数据集类型为 list, numpy ndarry 和 pandas.DataFrame
        estimator = KNeighborsClassifier(n_neighbors=5)
        estimator.fit(x_tra,y_train)  训练数据 x 必须 2 维， 标签值 y 必须 1 维。 训练
        estimator.predict(x_tes)       预测
        estimator.score(x_tes,y_test)       评估
        joblib.dump(estimator,'test.pkl')     保存模型
        estimator = joblib.load('test.pkl')    加载模型

    3.2 特征工程
        sklearn 转换器，特征工程的父类，实例化 Transformer， 调用 fit(x), transform(x), fit_transform(x) 方法
        a. 特征提取
            将任意数据（如文本或图像转）换为可用于机器学习的数字特征
            i. 字典特征提取 （将字典转换为特征向量，将字符串类别特征列进行转译，默认 one-hot 编译）
                from sklearn.feature_extraction import DictVectorizer
                d = [{'city': 'Shanghai', 'temperature': 30}, {'city': 'Shenzhen', 'temperature': 32},
                    {'city': 'Beijing', 'temperature': 25}]
                obj = DictVectorizer(sparse=True)    默认返回稀疏矩阵，非 0 项坐标 和对应值， 减少内存消耗
                m = obj.fit_transform(d)      d 为字典或者包含字典的迭代器，返回sparse矩阵
                    使用字典中所有离散特征（独特字符串）作为新的特征列名，并统计在所有字典中的出现次数，其他特征值不做变化，再转为稀疏矩阵
                    稀疏矩阵 sparse = True 时, m 的值为    (0, 1)	1.0     第 0 个字典包含特征列名列表中第 1 个词， 对应的出现次数为1
                                                        (0, 3)	30.0
                                                        (1, 2)	1.0
                                                        (1, 3)	32.0
                                                        (2, 0)	1.0
                    非稀疏矩阵 sparse = False 时, m 的值为    array([[ 0.,  1.,  0., 30.],
                                                                [ 0.,  0.,  1., 32.],
                                                                [ 1.,  0.,  0., 25.]])
                    或者使用 m.toarray() 方法获得非稀疏矩阵
                x2 = obj.inverse_transform(m)
                obj.get_feature_names_out()  获取特征列名 ['city=Beijing', 'city=Shanghai', 'city=Shenzhen','temperature']
            ii. 文本特征提取（使用单词作为特征）
                方法1： CountVectorizer 统计每个样本特征词的出现次数
                    from sklearn.feature_extraction.text import CountVectorizer
                    d = ["Some sentences 1",  "I like some some."]
                    import jieba      中文使用 jieba 分词（添加空格）
                    d2 = ["我爱北京天安门", "我去过北京。"]
                    l = list(jieba.cut("我爱北京天安门"))      ['我', '爱', '北京', '天安门']
                    d = [ " ".join(list(jieba.cut(s))) for s in d2]     ['我 爱 北京 天安门', '我 去过 北京 。']
                    trans = CountVectorizer(stop_words=None)    默认返回稀疏矩阵，非 0 项坐标 和对应值， 减少内存消耗
                        默认使用空格作来分割特征词， stop_words=['if','to'] 将 if, to 不作为特征词
                    m = trans.fit_transform(d)      d 为文本或包含文本字符串的可迭代对象，返回sparse矩阵
                    统计每个字符串出现特征词次数，再转为稀疏矩阵
                         m 的值为    (0, 2)	1    第 0 句含有特征词列表第 2 个单词，出现 1 词
                                    (0, 1)	1
                                    (1, 2)	2
                                    (1, 0)	1
                        m.toarray() 的值为   [[0, 1, 1],
                                            [1, 0, 2]]

                    x2 = obj.inverse_transform(m)
                    obj.get_feature_names_out()  标点，数字，字母不在特征词列表 ['like' 'sentences' 'some']

                方法2：TfidfVectorizer 找到只在这篇文章中出现概率高的（重要的）特征词
                    tf (term frequency) 词频，表示一个词在当前文档中出现的概率（出现次数 / 文章字数）
                    idf (inverse document frequency) 逆向文档频率，是一个词语普遍重要性的度量，log(总文件数 / 包含该词语的文件数)
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    trans = TfidfVectorizer(stop_words=None)    默认返回稀疏矩阵，非 0 项坐标 和对应值， 减少内存消耗
                        默认使用空格作来分割特征词， stop_words=['if','to'] 将 if, to 不作为特征词
                    m = trans.fit_transform(d)      d(x_train) 为文本或包含文本字符串的可迭代对象，返回sparse矩阵
                    统计每个字符串出现特征词次数，再转为稀疏矩阵
                         m 的值为    (0, 2)	1
                                    (0, 1)	0.8148024746671689   第 0 句中，特征词列表第 1 个单词的重要程度为 0.81
                                    (0, 2)	0.5797386715376657
                                    (1, 0)	0.5749618667993135
                                    (1, 2)	0.8181802073667197
                        m.toarray() 的值为   [[0.,     0.81480247, 0.57973867],
                                            [0.57973867,   0.        , 0.81818021]])
                    x2 = trans.inverse_transform(m)
                    trans.get_feature_names_out()  标点，数字，字母不在特征词列表 ['like' 'sentences' 'some']

        b. 特征预处理
            通过一些转换函数将特征数据转换成更适合算法模型的特征数据的过程。常见的有数据的无量纲化（归一化，标准化），将数据转为同一规格，解决
                特征单位，大小或方差比其他特征大很多，容易影响（支配）目标结果的问题。
            归一化（鲁棒性差，容易受到异常值影响）
                通过对原始数据进行变换把数据映射到[0,1]  x' = (x- min(x)) / (max(x) - min(x))  x" = x'*(hx-lx)+lx
                    hx, lx: 为转换后的最大值最小值，默认为 1,0
                trans = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)
                x_reg = trans.fit_transform(x)     x 可以是列表，ndarray，DataFrame 等，注意不要包括目标列
                    对于测试集应该使用 sc.transform(x)， 否则导致对测试集的学习（数据泄露）
            标准化 (鲁棒性好，不受异常值影响)
                通过对原始数据进行变换把数据映射到均值为 0， 标准差为 1 的范围内。 x' = (x - avg(x)) / std(x)
                trans = sklearn.preprocessing.StandardScaler()
                x_reg = trans.fit_transform(x)    x 为二维数组

        c. 特征降维
            在限定条件下，对于二维数组降低随机变量（特征）个数，得到一组“不相关”主变量的过程。相关特征（correlated feature）会造成冗余，
                降低算法性能。
                i. 特征选择：从所有特征（包含冗余或相关特征）中选择一个子集（主要特征）作为模型的训练数据。
                    Filter（过滤式）：寻找特征本身特点，特征与特征和目标之间的关联
                        方差选择法： 方差小的特征，可以认为是不重要的特征
                            trans = sklearn.feature_selection.VarianceThreshold(threshold=0.0)  删除所有方差低的特征
                            x_remain = trans.fit_transform(x)    x 为二维数组
                        相关系数法：特征与特征之间的相关程度。
                        皮尔森相关系数（Pearson Correlation Coefficient）： 衡量两个变量之间线性相关程度
                            r = [n * sum(xy) - sum(x) * sum(y)] / [sqrt(n * sum(x^2) - sum(x)^2) * sqrt(n * sum(y^2) - sum(y)^2)]
                                两个变量 x, y， 样本个数 n， r ∈ [-1,1]  r>0 正相关， r<0 负相关， r=0 无关, |r| 越大，相关性越强
                            from scipy.stats import pearsonr
                            r, p = pearsonr(x, y)   r 相关系数， p-value 值  x，y 为两列值
                            对于特征相关性高的特征，可以通过以下方法降维：
                                选择其中一个特征，删除其他
                                对于这些进行特征加权求和
                                主成分分析
                    Embedded（嵌入式）： 算法自动选择特征
                        决策树：信息熵，信息增益
                        正则化： L1， L2
                        深度学习：卷积

                ii. 主成分分析（PCA）：
                    通过线性变换将原始的（可能存在相关性的）高维特征数据转换为新的低维无相关性的特征数据的过程。在损失少量信息的情况下实现数
                    据压缩（降维）
                    from sklearn.decomposition import PCA
                    trans = PCA(n_components=0.95)        n_components 小数保留百分之多少信息，整数为减少到多少特征
                    x_remain = trans.fit_transform(x)     x 为二维数组


    3.3 K-近邻（KNN）算法  （根据邻居判断类别）
        如果一个样本在特征空间中的k个最相似(即特征空间中最邻近，通常使用欧式距离)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
        使用前必须对数据使用数据的无量纲化，适用于小数据场景（几千~几万）。
        简单有效，（没有构建模型）训练时间短，适合样本数多，共性多的数据。预测计算量大，时间长（惰性学习），预测解释性不强，不擅长不均衡的样本
        （需要重新采样，或者对距离使用权值），必须指定 k 值。

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=6, algorithm='auto',metric='minkowski',p=2)      # 默认参考 5 个邻居
            *** k 值选择： 选择奇数，过小容易收到异常点影响（过拟合），过大受到样本均衡的问题（欠拟合）。
            默认使用闵可夫斯基距离（p=2） 对应的欧式距离
            algorithm: auto, ball_tree, kd_tree, brute
        各种距离
            欧式距离（Euclidean Distance，直线距离）: d_12 = sqrt(sum_k((x_1k - x_2k)^2)))
            曼哈顿距离（Manhattan Distance，只能直行，横行）: d_12 = sum_k(abs(x_1k - x_2k))
            切比雪夫距离（Chebyshev Distance，只能直行，横行，45倍角斜行）: d_12 = max_k(abs(x_1k - x_2k))
            闵可夫斯基距离（Minkowski Distance，是一组距离的定义）: d_12 =(sum_k(abs(x_1k - x_2k)^p)) ** (1/p)
                p=1: 曼哈顿， p=2: 欧式， p -> inf: 切比雪夫
                缺点： 将各个分量的量纲（scale）/单位，相同的看待，且未考虑各个分量的分布可能是不同的。
                标准式欧式距离（将各个分量标准化）：sqrt(sum_k(((x_1k - x_2k)/s_k)^2)))   s_k 为第 k 维的标准差
            余弦距离（Cosine Distance，向量夹角余弦）: d_12 = cos(x_1, x_2) = sum_k(x_1k * x_2k) / (sqrt(sum_k(x_1k^2)) * sqrt(sum_k(x_2k^2))
            汉明距离（Hamming Distance，将一个变为另一个所需要的最小替换步数）: d_12 = sum_k(x_1k != x_2k)
            杰拉德距离（Jaccard Distance，两个集合的交集在并集中所占比例）: d_12 = (A∪B-A∩B)/(A∪B)
            马式距离（Mahalanobis Distance 通过样本分布计算）: d_12 = sqrt((x-mean(x))^T Cov^{-1} (x-mean(x)))
                其中 Cov 为协方差矩阵， 马氏距离排除变量之间的相关性干扰，建立在总体样本的基础上，要求总体样本数大于样本的维度。
        对于基本的线性搜索（每个点都计算与其他点的距离），对于 D 个特征的数据集，算法复杂度为 O(DN^2)
        kd 树: 把距离信息保存在二叉排序树中，使得搜索时间复杂度降低到 O(DNlog(N))。基本原理是：如果 A 和 B 距离很远， B
            和 C 距离很近，那么 A 和 C 距离也很远。
            构造 kd 树，选取中间的节点（所在方差最大的维度两侧节点数相同或差1），递归将数据空间通过此节点使用超平面将样本空间切分成两组（下一
            轮选取维度与这轮不同）对应左子树和右子树，直到叶子节点（空间中无数据点）。查找（判断新数据点位置）类似二分查找，依次和树的当前节点
            的切分维度的值对比，小则进入左子树，大则进入右子树，直到叶子节点，经过的节点记入栈中。计算从需判断点 g 到该叶子点的距离 r，再以
            g 为圆心, r 为半径判断圆是否与之前栈节点 a 的切分超平面相交。如果不相交则 a 出栈。如果相交，记录 a 到 g 的距离 r'，如果比
            r' < r，则更新 r = r'，a 出栈，同时将 a 另一侧的子树根节点入栈。直到栈中没有节点。
        ball tree: 克服 kd 树特征值高维失效问题（使用超球体）分割样本空间

    3.4 模型选择与调优
        a.（n 折）交叉验证
            将训练数据分为训练和验证集，将数据分为 n 份，其中1份作为验证集，其余为训练集。经过 n 轮测试，每轮更换验证集，获得 n 组模型结果
            取平均作为最终结果。 不能提高模型准确率，但使评估模型更准确可信。

        b. 网格搜索：对于超参数（需要手动指定）可选值的所有组合使用交叉验证进行评估，选出最优组合来建立模型。
            grid = {‘param_name’: [‘param_value1’， ‘param_value2’]}
            est = sklearn.model_selection.GridSearchCV(estimator, param_grid=None, cv=None, n_jobs=1)
                param_grid 估计器参数字典， cv 几折交叉验证,   n_jobs -1 使用全部多核 cpu
            est.fit(x,y)
            est.score(x,y)
            est.best_params_： 最好的参数
            est.best_score_： 最好的结果
            best_score_：最好结果
            best_estimator_： 最好参数的模型
            cv_results_： 每次交叉验证的验证集好训练集准确率

    3.5 朴素贝叶斯算法  （独立条件下的贝叶斯）
        朴素贝叶斯算法是生成式模型，通过先验概率和条件概率计算后验概率，进而求解最大后验概率对应的类别。
        基于数学理论，有稳定的分类效率，对缺失值不太敏感，算法较简单，分类准确度高，速度快。但是由于使用特征独立假设，特征有关联时效果不好。 常用于文
        本分类 （假设单词之间相互独立）

        概率： 一件事发生的可能性， P(X) ∈[0,1]
        联合概率：多个条件同时成立的概率， P(A,B)
        条件概率：事件 A 在另外一个事件 B 已经发生条件下的发生概率， P(A|B)
        相互独立： P(A,B) = P(A) * P(B) <==> 事件 A 与 B 相互独立。

        贝叶斯公式 P(A|B) = P(B|A) * P(A) / P(B)  A: 类别， B： 特征    P(B|A) = B特征在 A 类别出现的次数 / 类别 A中所有特征出现的次数和
            拉普拉斯平滑系数： 防止概率为 0， P(B|A) = (B特征在 A 类别出现的次数 + α) / (类别 A中所有特征出现的次数和 + α * 特征总数)
                α 为指定的平滑系数，一般为 1
        朴素： 假设特征与特征之间相互独立，即 P(A,B) = P(A) * P(B)
        P(A,B | C) = P(A | C) * P(B | C)

        from sklearn.naive_bayes import MultinomialNB
        est = MultinomialNB(alpha=1.0)
        est.fit(x_train, y_train)

    3.6 决策树
        决策树： 基于树结构进行决策的算法，根据数据特征是否满足节点特征分割条件进行划分，使用信息增益（为分类系统提供更多信息）多的特征来划分上层节
        点（ID3 决策树），从而减小信息熵，C4.5 使用信息增益比， CART 使用基尼系数。
        简单的理解，解释性强，树可以可视化。过于复杂的树容易过拟合。
        信息： （香农定义） 消除随机不定性的东西
        信息熵： 用于衡量信息量  H(X) = -sum_i{1~n} [P(x_i) * log_2 P(x_i)]  单位比特，最小为0    i 为类别
        信息增益： 特征 A 对训练数据集 D 的信息增益 g(D,A)， 定义为集合 D 的信息熵 H(D) 和特征 A 给定条件下 D 的条件熵 H(D|A) 之差，即
            g(D,A) = H(D) - H(D|A)  表示得知特征 A 的信息使得不确定性（信息熵）的减少程度
                H(D) = -sum_i{1~n} [(D_i / D) * log_2 (D_i / D)]   D_i 为属于 i 类别的样本个数，D 为样本总数
                条件熵 H(D|A) = sum_j{1~m} [(A_j / D) * H(A_j)]          A_j 为特征 A 的值 为 j 的样本个数
                H(A_j) = -sum_i{1~n} [(d_ij / A_j) * log_2 (d_ij / A_j)]    d_ij 为属于 i 类别特征 A 的值 为 j 的样本个数
        from sklearn.tree import DecisionTreeClassifier, export_graphciz
        est = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=None)    criterion='entropy' 使用信息熵
        est.fit(x_train, y_train))
        export_graphciz(est, out_file='tree.dot', feature_names=["occupation","age","income"])
        在 http://webgraphviz.com 显示导出的树文件

    3.7 随机森林
        集成学习：通过建立多个模型独立学习和预测，最后结合成组合预测。
        随机森林： 包含多个决策树的分类器，预测结果为所有决策树输出类别的众数。
            随机选择特征（抽取个数远小于特征总数，降维），随机选择训练集（bootstrap 随机有放回抽样，同一个数据可以被抽到多次），来产生不同的决策树。
        具有极好的准确率，处理高维数据不需要降维，能够评估特征重要性。
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, bootstrap=True,
            min_samples_split=2, min_samples_leaf=1,max_features="auto", random_state=None)
                max_features 可选值: "auto" -> sqrt(n_features) , "sqrt", "log2", None( = n_features)
        est.fit(x_train, y_train)

    3.8 线性回归  （不能解决（过/欠）拟合问题，需要无量纲化）
        a. 常见导数
            (常数)’ = 0
            (x^a)’ = ax^(a-1)
            (a^x)’ = a^x ln(a)
            (log_a(x))’ = 1/(x ln(a))
            (sin(x))’ = cos(x)
            (cos(x))’ = -sin(x)
            (u(x)+v(x))’ = u’(x)+v’(x)
            (u(x)*v(x))’ = (u’(x)v(x)-u(x)v’(x))/ (v(x))^2
            (g(h(x)))’ = g’(h)h’(x)
        b. 定义
            利用回归方程（函数）对一个或多个自变量（特征值）和因变量（目标值）之间关系进行建模的一种分析方式。
                公式： h(w) = w_1 * x_1 + w_2 * x_2 + … + b = w^T * x = xw    w = [[b],[w1],[w2]], x = [[1],[x1],[x2]]
            线性模型： 特征值与目标值之间建立的关系，一个特征值称为单变量回归，多个特征值为多元回归。主要有两种模型：线性关系和非线性关系（高次方
                y = w_1*x_1 + w_2 * x_2^2）
        c. 线性回归的损失和优化
            求解模型权重和偏置（参数），使得模型能预测准确
            损失（目标 / 成本）函数：衡量真实值和预测值的误差。 J(θ) = sum_i(h(x_i) - y_i)^2 = (y - Xw)^2
            优化算法：正规方程，梯度下降
            正规方程: 直接求得最优值 （特征多时速度慢且不一定能得到结果 O(n^3)，只能在线性回归使用，不能解决过拟合问题）
                w = (X^T X)^(-1) * X^T * y
                    2(Xw-y)*X = 0 ==> 2(Xw-y)*(X X^T)=0 ==> 2(Xw-y)*(X X^T)(X X^T)^-1 = 0 ==> Xw = y
                     ==>   X^T Xw = X^T y  (X^T* X 方阵确保可逆)  ==> (X^T X)^(-1)  X^T Xw = X^T X)^(-1) * X^T * y
                     ==> w = (X^T X)^(-1) * X^T * y
            梯度下降（Gradient Descent, GD）：计算所有样本值得到梯度。经过多次迭代，改进获得最优值 （计算量大，通用性强）
                W = W - α dcost/ dw
            随机梯度下降（Stochastic gradient descent, SGD）： 每次迭代仅考虑一个训练样本（速度快，但有很多超参数，对特征标准化敏感）
            小批量梯度下降法（mini-batch gradient descent）：使用随机的 m 个训练数据
            随机平均梯度（Stochastic average gradient, SAG）
        d. 实现
            from sklearn.linear_model import LinearRegression
            est =  LinearRegression(fit_intercept=True)   通过正规方程优化，fit_intercept 是否计算偏置
            est.fit(x_train, y_train)
            est.coef_  回归系数         est.intercept_   偏置

            from sklearn.linear_model import SGDRegressor   通过梯度下降优化
            est = SGDRegressor(loss='squared_loss', fit_intercept=True, penalty='l2',
                                alpha=0.0001, learning_rate='invscaling', eta0=0.01, max_iter=1000)
                squared_loss 最小二乘法。    learning_rate: "constant", "optimal", "invscaling"
                penalty='l2'  添加惩罚项等同于岭回归，只是采用 SGD
            est.coef_  回归系数         est.intercept_   偏置
        e. 回归性能评估
            均方误差（Mean Squared Error, MSE）: MSE = Sum_i{1~m}(y_i - mean(y))^2 / m , 也可作为损失函数，多了平均
            from sklearn.metrics import mean_squared_error
            mean_squared_error(y_true, y_pred)

    3.9 拟合
        欠拟合（）和过拟合（学习了太多针对训练集的特征）

        过拟合：在训练数据能获得更好的拟合，但是测试数据不能很好地拟合。学习了太多针对训练集的特征，模型过于复杂。需要减小高次项特征的影响，使用
            正则化。
            L1 正则化：L1 正则化使得一些权重系数直接为 0，删除特征影响（比较粗暴）   损失函数中添加惩罚项 + λ sum_j{1~n) abs(w_j)
            L2 正则化（更常用）：L2 正则化使得一些权重系数减小接近于0，削弱一些特征的影响。 在损失函数中添加惩罚项 + λ sum_j{1~n) w_j^2
        欠拟合：在训练数据和测试数据上都不能很好地拟合数据。学习到的特征太少，模型过于简单，区分太粗糙。需要增加数据，特征数量

        Lasso 采用 L1 正则项的线性回归
        Ridge 岭回归，是采用 L2 正则项的线性回归

    3.10 岭回归 （线性回归改进，需要无量纲化）
        岭回归：线性回归在建立回归方程时 加上了 L2 正则项的限制，解决过拟合。
        from sklearn.linear_model import Ridge   通过梯度下降优化
        est = Ridge(alpha=1.0, fit_intercept=True, solver="auto", normalize=False)
            alpha （λ） 为正则化力度  0~10     solver 在数据集，特征比较大时选择 SAG， normalize 同 StandardScaler
        est.coef_  回归系数         est.intercept_   偏置

    3.11 逻辑回归（需要无量纲化）
        逻辑回归：用于二分类问题，输出 0 或 1。但和回归算法之间有一定的联系。
        原理
            逻辑回归的输入就是线性回归的输出，然后通过 sigmoid 函数 （1 / (1+e^(-x))）将其映射到 0~1 之间，采用默认 0.5 为阈值，
            区分两个类。
            h(0) = 1 / (1 + e^(-0^T * x))
        损失与优化
            对数似然损失： cost(h_0(x),y) =  -log(h_0(x))  if y=1
                                       = -log(1-h_0(x)) if y=0
                cost(h_0(x),y) = sum_i{1~m} [-y_i * log(h_0(x)) - (1-y_i) * log(1-h_0(x))]   对于 m 个数据
            使用梯度下降进行优化
            from sklearn.linear_model import LogisticRegression
            est = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)   C 正则化系数， 默认 SAG 优化
            等同于 SGDClassifier(average=True)  默认随机梯度下降，添加average 使用 SAG
        分类的评估
            混淆矩阵： 行索引真实结果正反例， 列索引预测结果正反例。
            精确率（Precision）：预测结果为正例样本中真实为正例的比例   TP / (TP + FP)
            召回率（Recall）：真实为正例的样本中预测为正例的比例   TP / (TP + FN)
            F1-score: 反应了模型的稳健性  F1 = 2TP / (2TP + FN +FP) = 2 Precision * Recall / (Precision + Recall )
            from sklearn.metrics import classification_report
            rep = classification_report(y_true, y_pred,labels=None, target_names=None)
                label: 类别对应的数字， 返回每个类别的精确率和召回率
            在样本不均衡时，使用 ROC 曲线和 AUC 指标
                TPR: 所有真实类别为正例的样本中，预测为正例的比例 (同召回率)   TP / (TP + FN)
                FPR: 所有真实类别为反例的样本中，预测为正例的比例    FP / (FP + TN)
                ROC 曲线： x 轴 FPR 左到右 0 -> 1, y 轴 TPR 下到上 0 -> 1。 通过变换二分类的阈值（原先默认0.5）获得 ROC 曲线。
                    随机猜测 为 （0,0）到（1,1）直线 （TPR = FPR），AUC 为 0.5。 完美预测为（0,1）到 （1,1）直线，AUC 为 1， AUC 范围 [0.5,1]
                    from sklearn.metrics import roc_auc_score
                    roc_auc_score(y_true, y_pred)    y_true 反例必须为0， 正例为 1。

    3.12 K-Means 算法
        迭代式算法，简单实用。但是容易收敛到局部最优（可通过多次聚类取最佳值缓解）。
        步骤：
            1. 随机设置 k 个特征空间内的点作为初始的聚类中心。
            2. 对于其他每个点计算到 k 个中心点的距离， 未分类的点选择距离最近的聚类中心点作为标记类别。
            3. 对于（k 个）聚类中心，计算所有被标记为该类的点的（所有维度）平均值作为新的中心点
            4. 重复第二步，直到新中心点与原中心点一致（可以加入可容忍误差）。
        k 值选取： 看需求， 或者网格搜索调节超参数
            from sklearn.cluster import KMeans
            estimator = KMeans(n_clusters=3, init='k-means++'， max_iter=300)    中心个数，中心初始方法
            estimator.fit(X)
            estimator.cluster_centers_    聚类中心点
            estimator.labels_     训练后标记类型
            estimator.predict(new_X)
        聚类效果评估
            轮廓系数 (silhouette score)：  期望簇内距离小， 簇间距离大
                sc_i = (b_i - a_i) / (max(b_i, a_i))     b_i 为到其他簇群的所有样本距离最小值， a_i 为到本身簇的距离平均值。
                计算所有样本点的轮廓系数平均值。 轮廓系数取值范围 [-1,1] 越大聚类效果越好
            from sklearn.metrics import silhouette_score
            silhouette_score(X, estimator.labels_, metric='euclidean')

    3.13 神经网络
        不要初始化权重为 0 的矩阵，会导致训练缓慢，最好均匀分布的小数。

4 Pytorch
    Pytorch 是基于 Numpy 的科学计算包，能够使用 GPU 加速计算，提供灵活，高速的深度学习平台。
    Tensor (torch.Tensor)张量： 类似于 Numpy 的 ndarry 的数据结构， 区别在于可以利用 GPU 加速运算。

    import torch
    创建张量
        x = torch.empty(5,3)    未初始化的矩阵（数据为所占内存中没意义的数据）
        x = torch.rand(5,3)      随机初始化的矩阵，范围 [0,1) 均匀分布
        x = torch.zeroes(5,3, dtype=torch.long)     形状写成 (5,3) 也可以
        x = torch.tensor([2.5, 3.5])
        x2 = x.new_ones(3,2, dtype=torch.double)    创建与 tensor 相同的设备和数据类型；  _randn
        x = torch.rand_like(x) 用于创建一个与 tensor 形状相同的 tensor   randn, randint
        x.shape     x.size()         返回张量形状 torch.Size([3, 2])
        x.ndim                返回张量维度
    切片 (同 numpy)
        x[:, :3]
        x[1, :]

    x.view(-1,8)   改变张量形状， -1 自动匹配个数
    x[1,1].item()   返回张量中单个元素的标量值

    torch tensor (除了 CharTensor) 和 numpy ndarry 可以互相转换，共享底层的内存空间，改变其中一个值会同时改变另一个。
    x.numpy()       张量转换为 numpy ndarry
    x= torch.from_numpy(np.ones((2,3)))       numpy ndarry 转换为张量
    y = x + 1              张量加法，需要赋值返回值
    y = torch.add(x,y)
    np.add(a,1, out=a)   原地 ndarray 中所有元素加法
    torch.add(x,y, out=x)   原地张量中所有元素加法
    x.add_(y)     原地张量中所有元素加法

    移动张量到指定设备，需要在同一设备上执行数据操作
    if torch.cuda.is_available():        判断 cuda 是否安装，且 GPU 可用
        device = torch.device('cuda')     使用 GPU
        x.to(device)                      将张量移动到 GPU
        x.device                           device(type='cuda')
        y = torch.ones_like(x, device=device)     在 GPU 上创建张量
        (x+y).to('cpu', torch.double)      将张量移动到 CPU

    张量求导
        x2 = torch.ones(2, 2, requires_grad=True)      追踪张量变化用于求导（微分）
        x2.requires_grad_(True)       原地修改是否需要追踪求导
        x3 = x2 ** 2
        x3.requires_grad             True 由需要追踪求导张量推出的张量依然需要追踪求导   返回 Boolean
        x3.grad_fn    自定义张量返回 None, 否则返回最后的一个数据操作 （ex. <MulBackward0>）
        x3.grad_fn.next_function[0][0]   返回之前一个方法，可以多次使用
        x2.mean().backward()          对标量执行反向传播，将梯度计入各个张量的 grad 属性内
        x2.grad                返回梯度值
        y = x2.detach()            从计算图中撤出，在未来回溯计算中不会再求导，并返回张量
        with torch.no_grad():      在代码块中内容终止求导 （建议）
            x3 = x2 ** 2


    定义一个拥有可学习参数的神经网络
    遍历训练数据集
    处理输入数据使其流经神经网络
    计算损失值
    将网络参数的梯度进行反向传播
    以一定的规则更新网络的权重

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):     只支持 mini-batch 的输入， 不支持单个样本输入（需要 UNsqueeze(0) 增加维度）
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.fc1 = nn.Linear(16 * 6 * 6, 120)
            self.fc2 = nn.Linear(120, 84)
            self.pool = nn.MaxPool2d(2, 2)
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 3)    输入一维，输出六维，卷积核大小 3*3
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.fc1 = nn.Linear(16 * 6 * 6, 120)  三层全连接层，此处 6 根据图片尺寸经过卷积操作获得
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)     10 为分类数， pytorch 交叉熵损失函数已经包含 softmax 和负对数损失

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  对第一层卷积网络经过relu 激活函数后进行 (2,2)窗口大小的最大池化操作
            # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = self.pool(F.relu(self.conv2(x))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        def num_flat_features(self, x):
            size = x.size()[1:]    不包含第 0 维的 batch_size
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


    batch_size = 4
    transform = transforms.Compose(            torchvision 数据集为 PIL格式，值域 [0,255], 先转为 Tensor，再变为均值 0.5， 方差 0.5
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)   制作小批量数据集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = trainset.classes     # ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    net = Net()
    net.to(device)
    params = list(net.parameters())
    print(len(params), params[0].size())   params[0] 为第一层卷积层参数
        # target = torch.randn(1, 10)
        # target = target.view(1, -1)    改变目标为二维张量，和输出匹配
        # loss = nn.MSELoss(out, target)
    criterion = nn.CrossEntropyLoss()
    print(loss.grad_fn,  loss.grad_fn.next_functions[0][0].next_functions[0][0])
    optimizer = optim.SGD(net.parameters(), lr=0.01)  优化器， 传入参数为模型所有参数， 学习率

    for epoch in range(2):
        for i, data in enumerate(trainloader, 0):
            input, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()    梯度清零， 否则会累加之前批次的梯度
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()          算出参数梯度
            optimizer.step()        更新参数
            # print(net.conv1.bias.grad)

            total_loss += loss.item()     获取单个元素的张量的标量值
            if (i+1) % 1000 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 1000))
                total_loss = 0.0

    # 手动梯度下降， 不建议
    # for f in net.parameters():
    #     f.data.sub_(0.001 * f.grad.data)

    torch.save(net.state_dict(), './cifar_net.pth')
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    net = Net()
    net.load_state_dict(torch.load('./cifar_net.pth'))

    correct, total = 0, 0                               整体准确率
    class_correct, class_total = [0] * 10, [0] * 10     单类准确率

    with torch.no_grad():      测试过程不要追踪求导
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, axis=1)  选取最大的概率对应的类
            c = predicted.eq(labels).squeeze()     (predicted == labels).squeeze()
            # print('Predicted: ', ' '.join('%s' % classes[predicted[j]] for j in range(4)))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    x.means()
    x.eq(y).all()       数据结果比较
    torchvision.utils.make_grid(images)

5. 自然语言处理 （Natural Language Processing, NLP）
    是关注计算机与人类语言转换的领域，存在基于规则和统计两种方法，前者通过语法分析来处理（现以落伍），后者通过统计来处理。

    5.1 文本预处理
        a. 文本处理的基本方法
            i. 分词： 将连续的字序列按照一定的规范重新组合成词序列的过程。词作为语言语义理解的最小单元，是人类理解文本语言的基础， 也是 NLP
                领域高阶任务的基础。

                流行的中文分词工具 jieba， 支持多种分词模式， 包括： 精确模式， 全模式， 搜索引擎模式， 混合模式。支持繁体分词，用户自定义词典
                pip install jieba
                精确模式分词： 试图将句子最精确地切开，适合文本分析
                全模式： 把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义
                搜索引擎模式：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
                import jieba
                content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
                jieba.cut(content, cut_all=False)     # cut_all 默认 False（代表精确模式）， True (代表全模式)。返回一个生成器对象
                jieba.cut_for_search(content)        # 搜索引擎模式
                jieba.lcut(content, cut_all=False)    # 返回切分好的词列表    lcut_for_search
                    ['工信处','女干事','每月','经过','下属','科室','都','要','亲口','交代','24','口','交换机','等','技术性','器件','的','安装','工作']
                构造自定义词典， 适合识别专业词汇。每一行分三部分： 词语，词频（可省略）， 词性（可省略）， 空格隔开。例如：
                    云计算  5 n
                    李小福 2 nr
                    easy_install 3 eng
                    好用 30
                jieba.load_userdict("./userdict.txt")


                中英文分词工具 hanlp，基于 tensorflow 2.0，使用深度学习的技术
                pip install hanlp
                import hanlp
                tokenizer = hanlp.load('CBT6_CONVSEG')
                l = tokenizer("工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作")
                tokenizer = hanlp.utils.rules.tokenize_english    # 英语以空格切分
                l = tokenizer('Mr. Hanks bought a car for 1.5 thousand dollars.')

            ii. 词性标注：
            iii. 命名实体识别：是指识别文本中具有特定意义的实体，如人名、地名、机构名、时间、专有名词等。


        b. 文本张量表示方法
            文本张量： 将一段文本用张量进行表示，其中一般将词汇表示成向量，称为词向量，再由各个词向量按顺序组成矩阵形成文本表示。能够使语言文本
                可以作为计算机处理程序的输入，进行之后的解析工作。
            i. one-hot （独热）编码：每个词表示成具有 n 个元素的向量，仅有一个元素为 1，其余为 0。不同词汇 1 的位置不同，n 为不同词汇总数。
                操作简单，容易理解。但是割裂了词与词之间的联系（同为歌手的两个人名之间不存在任何联系），且在大语料集下，每个向量长度过大，占用
                大量内存。
                from sklearn.externals import joblib
                from keras.preprocessing.text import Tokenizer
                vocab = {"周杰伦"，“林俊杰”，“陈奕迅”}
                t = Tokenizer(num_words=None, char_level=False)
                t.fit_on_texts(vocab)

                for word in vocab:
                    zero_list = [0] * len(vocab)
                    one_index = t.texts_to_sequences([word])[0][0] - 1      texts_to_sequences 返回二维矩阵
                    zero_list[one_index] = 1
                    print(word, zero_list)
                joblib.dump(t,'./tokenizer')
                t = joblib.load('./tokenizer')

            ii. Word2vec：是一种将词汇表示成向量的无监督训练方法，将构建神经网络模型，将网络参数作为词汇的向量表示，包含 CBOW 和 skipgram
                两种训练模式。
                CBOW (continuous bag of words)：给定一段用于训练的文本语料，再选定某段长度（窗口）作为研究对象，使用上下文词汇（已知）预
                测窗口中的目标词汇（未知，在窗口正中间）。
                例： 一句话 Hope can set you free. 假设参数窗口大小为 3， 对于第一轮，输入 Hope， set 预测 can. Hope, set 为 5*1
                    的 one-hot 矩阵，与变化矩阵 W (3*5) 相乘再相加，得到 3*1 的隐藏层（上下文表示矩阵），让后用隐藏层乘以另一个变化矩阵
                    W1 (5*3)，得到 5*1 的结果矩阵，通过 softmax 与目标词汇 can 的 one-hot 矩阵进行 cross entropy 损失计算，并更新网
                    络参数完成一次模型迭代。
                    最后窗口每次向后移动一个单词，更新参数，直到语料遍历完成，得到最终的变化矩阵 W (3*5)，这个变化矩阵与每个词汇的 one-hot
                    矩阵（5*1）相乘，得到 3*1 的矩阵就是该词汇的 word2vec 张量表示。
                skipgram：给点一段用于训练的文本语料，再选定某段长度（窗口）作为研究对象，使用（窗口正中间）目标词汇预测上下文词汇。
                例： 一句话 Hope can set you free. 假设参数窗口大小为 3， 对于第一轮，输入 can， 预测 Hope，set。can (5*1) 的
                    one-hot 矩阵，与变化矩阵 W (3*5) 相乘，得到 3*1 的隐藏层（上下文表示矩阵），让后用隐藏层分别乘以变化矩阵 W1 (5*3)，
                    W2 (5*3)，得到两个 5*1 的结果矩阵，与目标词汇 Hope, set 的 one-hot 矩阵进行损失计算，并更新网络参数完成一次模型迭代。
                    之后与 CBOW 相同。

                wget http://mattmahoney.net/dc/enwik9.zip -P data
                unzip data/enwik9.zip -d data
                perl wikifil.pl data/enwik9 > data/fil9    去除 xml 标签对
                pip install git+https://github.com/facebookresearch/fastText.git
                import fasttext
                model = fasttext.train_unsupervised('data/fil9', 'cbow', dim=300, epoch=1, lr=)     训练 word2vec 无监督神经网络
                vec = model.get_word_vector("the")    获取 the 对应的词向量
                words = model.get_nearest_neighbors("the")    获取 the 对应的近邻词
                model.save_model("data/fil9.bin")    保存模型
                model = fasttext.load_model("data/fil9.bin")    加载模型

                from gensim.models import Word2Vec
                custom_corpus = [["I", "am", "a", "boy"], ["I", "like", "to", "play", "ball"]]
                model = Word2Vec(sentences=custom_corpus, vector_size=100, window=5, min_count=1, sg=0)
                    sentences 文本语料, 二维字符串列表。vector_size 转化后词向量大小。window 窗口大小。min_count 最低出现次数忽略阈值
                    sg 方法 (0 代表 CBOW, 1 代表 skip-gram).  一步建立词库，和训练
                    model.build_vocab(common_texts)
                    model.train([["I","am"]], total_examples=1, epochs=1)  最类似的单词
                model.save("word2vec.model")
                loaded_model = Word2Vec.load("word2vec.model")
                model.wv['ball']   转换单词为词向量
                model.wv.most_similar('ball', topn=5)
                model.wv.similarity('ball','boy')   返回两词相似度

            iii. Word Embedding 词嵌入: 将词汇映射到指定维度的空间， 广义包括所有密集词向量的表示方法（包含 word2vec），狭义指神经网络中
                加入的 embedding 层，对整个网络训练的同时产生的 embedding 矩阵(对应 word2vec 中的矩阵 W)，是训练过程中所有输入词汇的词向
                量组成的矩阵

                词嵌入可视化
                import torch
                from torch.utils.tensorboard import SummaryWriter
                import fileinput
                with SummaryWriter() as writer:
                    embedded = torch.randn(100,50)  100 个词汇， 50 维向量
                    meta = list(map(lambda x: x.split(' '), fileinput.FileInput("./vocab100.csv")))
                    writer.add_embedding(embedded, metadata=meta)

                tensorboard --logdir runs --host 0.0.0.0

        c. 文本语料的数据分析： 能够帮助理解数据语料，检查语料可能存在的问题，指导之后模型训练工程中一些超参数的选择。
            i. 标签数量分布 (确保每类数据个数相近，否则需要进行数据增多或者删减)
                import seaborn as sns
                plt.style.use('fivethirtyeight')
                train_data = pd.read_csv('./data/cn_data/train.tsv', sep='\t')
                sns.countplot('label', data=train_data)
                plt.show()

            ii. 句子长度分布
                模型的输入要求固定尺寸的张量，合理的长度范围对之后进行句子截断补齐（规范长度）起到关键指导作用。
                train_data['sentence_length'] = list(map(lambda x:len(x), train_data['sentence']))
                sns.countplot('sentence_length', data=train_data)
                sns.distplot(train_data['sentence_length'])       histogram
                sns.stripplot(x='sentence_length', y='label', data=train_data) 散点图用于发现异常点

            iii. 词频统计与关键词词云 （可视化出现的词，高频词字体大，可以对当前语料质量进行评估，对违法语料变迁含义的词汇进行人工审核修正）
                import jieba
                from itertools import chain    扁平化列表
                train_p_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data[train[data['label']==1]['sentence'])))
                print(train_p_vocab, len(train_vocab))    获得正样本出现过的不重复的词汇

                import jieba.posseg as pseg
                from wordcloud import WordCloud
                adj_word = [g.word for g in pseg.lcut(train_p_vocab) if g.flag == 'a']  所有形容词列表
                    word 属性为切分的词， flag 为词性
                wordcloud = WordCloud(font_path='simhei.ttf', max_words=100, background_color='white', max_font_size=80)
                wordcloud.generate(" ".join(adj_word))
                plt.figure()
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.show()

        d. 文本特征处理： 为语料添加具有普适性的文本特征，增加模型评估指标。以及对语料进行必要的处理，如长度规范。
            i. 添加 n-gram 特征：给定一段文本序列，其中 n 个相邻的词共同出现的独特排列作为特征，大多使用 bigram，trigram。
                n = 2
                set(zip(*[input_list[i:] for i in range(n)])   对列表进行错位复制，使用 zip 组合并去重

            ii. 文本长度规范: 一般模型输入需要等尺寸大小的矩阵，因此需要对文本数值映射后的长度进行规范，根据句子长度分布选择覆盖绝大多数文本的
                合理长度，对超长文本进行截断，对不足文本进行补齐（一般使用数字 0）
                from keras.preprocessing.sequence import pad_sequences
                x_train = pad_sequence(x_train, 100)    100 为文本截断长度（设置为覆盖90%左右的语料的最短长度）
                    超长的文本张量，取最后的 100 个词，不足 100 的文本在前面补 0。

        e. 数据增强方法
            i. 回译数据增强法: 一般基于谷歌翻译接口，将文本数据翻译成另一种语言（一般选择小语种），之后再翻译回原语言，获得与原语料同标签的新语
            料，增加到原数据集中认为是对原数据集的数据增强。
            操作简便，获得新语料质量高。短文本语料回译可能存在较高重复率，并不能增大样本的特征空间，可进行多次连续翻译（不超过3次，否则失真）
            from googletrans import Translator
            corpus = ["难吃死了","味道还不错"]
            translator = Translator()
            translations = translator.translate(corpus, dest='ko')  翻译成韩文
            ko_result = [t.text for t in translations]
            translations = translator.translate(ko_result, dest='zh-cn')   回译成中文
            corpus_extended = [t.text for t in translations]

    5.2 经典的序列模型
        a. HMM (Hidden Markov Model) 隐马尔可夫模型，用来描述一个含有隐含未知参数的马尔可夫过程，其目的是根据有限的观察数据，去
            推测一个隐藏的、未知的或不可观测的参数。一般以文本数据为输入，以该数据的隐含序列（各个单元中存在具有关联的隐性信息组成的序列，例如原
            数据（观测）序列的词性序列）为输出。用来解决文本序列标注问题，如分词，词性标注，命名实体识别等。
            HMM 模型表示为 λ = HMM(A,B,π)， A 为状态转移概率矩阵，B 为观测概率矩阵，π 为初始状态概率向量。训练需要观测序列及其对应的隐含序
            列，通过极大似然估计或贝叶斯估计得到参数，使由观测序列到对应隐含序列的概率最大。为了简化计算，模型使用了隐含假设：隐含序列中的每个单
            元的可能性只与上一个单元有关。之后，对于输入观测序列 (x1, x2,...,xn)，根据训练得到的A,B,π，计算对应隐含序列的条件概率分布。最后，
            使用维特比算法从隐含序列的条件概率分布中找出概率最大的一条序列路径，即为该观测序列对应的隐含序列 (y1, y2,...,yn)。

        b. CRF (Conditional Random Field) 条件随机场，一般以文本序列数据为输入，以该序列对应的隐含序列为输出。用来解决文本序列标注问题，
            如分词，词性标注，命名实体识别等。
            CRF 模型表示为 λ = CRF(w1,w2,...,wn)。训练需要观测序列及其对应的隐含序列，使用人工特征工程不断训练得到一组参数，使由观测序列到
            对应隐含序列的概率最大。训练完后，对于输入序列 (x1, x2,...,xn)，根据训练得到的 w1, w2,... 计算对应隐含序列的条件概率分布。最后，
            使用维特比算法从隐含序列的条件概率分布中找出概率最大的一条序列路径，即为该观测序列对应的隐含序列 (y1, y2,...,yn)。
            CRF 没有隐马假设，所以计算较慢。但如果实际不满足假设，HMM 准确率大大降低

        近年来，经典模型被深度学习所替代。

    5.3 循环神经网络 RNN（Recurrent Neural Network）
        循环神经网络：一般以序列数据为输入，通过网络内部的结构设计捕捉序列之间的关系特征，一般输出也是序列。对于第 i 层，隐藏层的输出为：
        h_t = σ_h (W_h · x_i + U_h · h_t-1 + b_h)，或者表示为 h_t = σ_h (W_t · [x_i, h_t-1] + b_h), 其中 [x_i, h_t-1] 为输入
        和隐藏矩阵的拼接。 输出层为 y_i = σ_y (W_y · h_i + b_y)。 其中 W, U, b 为可学习的参数。h_0
        为初始隐藏层，一般为全零向量。 σ_h 一般为 tanh 函数，σ_y 一般为 softmax 函数。损失函数为序列每步损失之和
        RNN 结构很好利用序列之间的关系，广泛应用连续的输入序列的问题（语言，语音，文本分类，情感分析，机器翻译等），在很短序列中的效果比 LSTM,
        GRU 好。内部结构简单，计算较快，参数总量少，在短序列（10 以内）任务上表现好。对于长序列，反向传播存在梯度消失（权重无法更新，训练失败），
        梯度爆炸（结果溢出 NaN），导致效果差。RNN, LSTM, GRU 不能并行计算，训练慢。
        RNN 模型的分类：
            按输入和输出结构： N vs N, N vs 1，1 vs N, N vs M (seq2seq, 第一部分编码器没有输出层，最后第 N步 隐藏层输出一个变量 c，
                作为第二部分解码器输入层的输入作用在每一步，返回一个长度为 M 的序列)
            按内部构造： 传统RNN， LSTM， Bi-LSTM，GRU，Bi-GRU

        import torch.nn as nn
        rnn = nn.RNN(5,6,1)    输入张量尺寸，隐藏层尺寸（神经网络节点个数），隐藏层层数
        input = torch.randn(1,3,5)    输入序列长度，batch_size, 输入张量尺寸
        h0 = torch.randn(1,3,6)      隐藏层层数（双向 RNN *2），batch_size，隐藏层尺寸
        output, hn = rnn(input, h0)   返回最后一层输出层和隐藏层的输出值，调用父类 Module __call__ 方法
        print(output.shape, input.shape)   (1,3,6),(1,3,6)   两个值相等

    5.4 长短时记忆结构 LSTM (Long Short Term Memory)
        是传统 RNN 的变体，能够有效捕捉长序列之间的语义关联，缓解梯度消失和梯度爆炸（不能完全解决）。结构更复杂，包括遗忘门，输入门，相关（细胞
        状态）门，和输出门。 其中 x 为输入矩阵，输出 c 为长期记忆输入矩阵，输出 h 为短期记忆输入矩阵（最后阶段作的 h 作为模型输出值），训练参数
        为 b 偏执，和 W 参数矩阵。⊙ 表示矩阵元素一对一相乘。
        f_t = σ(W_f · [h_t-1, x_t] + b_f)  遗忘门输出  σ： sigmoid 函数，与长期记忆相乘，决定了其保留比例
        i_t = σ(W_i · [h_t-1, x_t] + b_i)   输入门输出，决定了细胞中间状态值的保留比例
        c'_t = tanh(W_c · [h_t-1, x_t] + b_c)               相关（细胞状态）门（取决于短期记忆）
        c_t = f_t ⊙ c_t-1 + i_t ⊙ c'_t                     长期记忆，没有和权重相乘导致的长序列梯度消失爆炸问题
        o_t = σ(W_o · [h_t-1, x_t] + b_o)                 输出门
        h_t = o_t ⊙ tanh(c_t)                             输出 (新的短期记忆)

        Bi-LSTM： 双向LSTM，不改变内部结构只是将输入序列复制一份，分别做正向和反向计算，最后将两个（对应同一输入序列步数据的）输出拼接在一起作
            为输出。能捕捉到语言语法中定的前置后置特征，增强语义关联，但复杂度增加一倍。

        import torch.nn as nn
        lstm = nn.LSTM(5,6,2)    输入张量尺寸，隐藏层尺寸（神经网络节点个数），隐藏层层数
        input = torch.randn(1,3,5)    输入序列长度，batch_size, 输入张量尺寸
        h0, c0 = torch.randn(2,3,6), torch.randn(2,3,6)      隐藏层层数（双向 RNN *2），batch_size，隐藏层尺寸
        output, (hn,cn) = lstm(input, (h0,c0))   返回最后一层输出和 短期,长期记忆输出值
        print(output.shape, hn.shape,cn.shape)   (1,3,6), (2,3,6), (2,3,6)  output 与 hn[1] 相等

    5.5 GRU（Gated Recurrent Unit）
        门控循环单元，是传统 RNN 的变体， LSTM 简化版，结构较简单，也能有效捕捉长序列之间的语义关联。结构包括更新门和重置门。其中 x 为输入矩阵,
        输出 h 为短期记忆输出矩阵
        z_t = σ(W_z · [h_t-1, x_t] + b_z)               更新门（决定当前时刻的短期记忆是否更新）
        r_t = σ(W_r · [h_t-1, x_t] + b_r)               重置门（决定长期记忆是否更新）
        h'_t = tanh(W_h · [r_t ⊙ h_t-1, x_t] + b_h)
        h_t = (1 - z_t) ⊙ h_t-1 + z_t ⊙ h'_t             输出

        Bi-GRU： 双向 GRU, 原理同 Bi-LSTM

        import torch.nn as nn
        gru = nn.GRU(5,6,2)    输入张量尺寸，隐藏层尺寸（神经网络节点个数），隐藏层层数
        input = torch.randn(1,3,5)    输入序列长度，batch_size, 输入张量尺寸
        h0 = torch.randn(2,3,6)   隐藏层层数（双向 RNN *2），batch_size，隐藏层尺寸
        output, hn = gru(input, h0)   返回最后一层输出和 短期,长期记忆输出值
        print(output.shape, hn.shape)   (1,3,6), (2,3,6)  output 与 hn[1] 相等

    5.6 注意力机制
        注意力机制，是自然语言处理中一种重要的信息处理方式，其核心思想是让模型更加关注输入序列中与当前任务相关的部分。是注意力计算规则应用的深度
        学习网络的载体。包括全连接层和相关张量处理，使其与应用的网络融为一体。在 NLP 中，大多使用 seq2seq 架构（编码器解码器），注意力机制使
        解码器每一步输出时都可以查看编码器所有步的输出，并计算权重使编码器某些输出更重要。解决了编码器单个输出无法涵盖序列所有信息的问题。
        在编码器中，注意力机制解决表征问题，相当于特征提取过程，一般使用自注意力（self-attention）在解码器中，注意力机制能根据模型目标有效的聚
        焦编码器的输出结果作为输入，提升解码器的效果。改善以往编码器输出是定长张量，无法存储过多信息的。
        注意力机制解决 seq2seq 序列过长导致学习不到开头内容，但是计算相关性复杂度 O(mn)， m,n 为编码器，解码器序列长度，需要极大算力。

        注意力计算规则： 对于输入 Q(query, 序列), K(key 关键词), V(value), 计算 Q 与 K 的相似度，相似度越高，V 越有可能被选中，结果代表
            query 在 key 和 value 作用下的注意力表示。当 Q=K=V，时，称作自注意力计算规则。
        常用规则：（相关性计算）
            Attention(Q,K,V) = Softmax(Linear([Q,K]))·V      [Q,K] 为两个张量纵轴拼接，最常用
            Attention(Q,K,V) = Softmax(sum(tanh(Linear([Q,K]))))·V    Q： h   K: s
            Attention(Q,K,V) = softmax(Q·K^T/sqrt(d_k))·V       d_k 为 k 最后一维的大小

        例如， 模式 a: Q 为解码器隐藏层, K 编码器隐藏层, V 编码器隐藏层 (先调用 RNN,再 Attention )
             模式 b:  Q 为解码器(embedded)输入, K 解码器器隐藏层, V 编码器隐藏层 (先调用 Attention ,再 RNN)
        step 1: 源语句 < s > i am fat < e > 输入 Encoder 进行编码，每一步都得到对应输出 encoder outputs（图中浅蓝色长条）；
        step 2a: 进入解码阶段，首步输入 < s >，得到相应表征 hidden state（图中的黄色长条）；
        step 2b: 进入解码阶段，首步输入 Embedded 后的 < s >
        step 3a: 将 hidden state 分别与 encoder outputs 进行内积计算，并通过 softmax 转换为概率形式，这里称为 weights，此概率代表了 hidden state 与各 encoder outputs 间的相关程度；
        step 3b: 将 < s > 分别与 hidden state 进行内积计算，并通过 softmax 转换为概率形式，这里称为 weights，此概率代表了  < s > 与各 hidden state 间的相关程度；
        step 4: 将 weights 分别与 encoder outputs 相乘再相加，得到加权相加后的向量表征（图中的绿色长条），称为 weighted context;
        step 5a: 将 weighted context(如果不是第三种规则，则需要与 hidden state 拼接)，得到词典大小的维度，softmax 预测下一个单词；
        step 5b: 将 weighted context(如果不是第三种规则，则需要与 < s >  拼接)，全连接映射到 < s > 大小，作为输入传入 RNN， softmax 预测下一个单词；
        step 6: 将下一个单词输入解码单元，重复 step 2-step 5，直至达到最长长度或者预测出终止符 < e > 。

        当注意力权重矩阵和 V 都是三维张量且第一维代表 batch_size 时，则做 bmm 运算（特殊的张量乘法）
        a,b = torch.randn(10,3,4), torch.randn(10,4,5)
        torch.bmm(a,b)     返回形状 (10,3,5)，后两维进行矩阵乘法

        # Attention(Q,K,V) = Softmax(Linear([Q,K]))·V
        import torch.nn.functional as F
        class Attn(nn.Module):
            def __init__(self, query_size, key_size, value_size1,value_size2, output_size):
                # query_size: query 最后一维大小， key_size： key 最后一维大小,   value_size1: value 倒数第二维大小,
                # value_size2: value 最后一维大小， output_size: 输出最后一维大小
                super(Attn, self).__init__()
                self.query_size = query_size
                self.key_size = key_size
                self.value_size1 = value_size1
                self.value_size2 = value_size2
                self.output_size = output_size

                self.attn = nn.Linear(self.query_size + self.key_size, self.value_size1)
                self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)

            def forward(self, Q, K, V):
                attn_weights = F.softmax(self.attn(torch.cat((Q[0],K[0]),1), dim=1)  将 Q,K 纵轴拼接，经过线性变化和softmax
                attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)
                output = torch.cat((Q[0], attn_applied[0]), 1)     如果是第三种点积，使用自注意力，则不需要与 Q 拼接
                output = self.attn_combine(output).unsqueeze(0)   经过线性变化，改变尺寸。输出需要三维张量
                return output, attn_weights

        attn = Attn(32,32,32,64,64)
        output = attn(torch.randn(1,1,32),torch.randn(1,1,32),torch.randn(1,32,64))    返回 (1,1,64)  (1,32) 矩阵

    5.7 teacher_forcing
        是一种用于序列生成任务的训练技巧，改变上一步出错的情况，因为训练时我们是已知正确的输出应该是什么，因此可以强制将上一步结果设置成正确的输
        出，防止使用错误结果作为输入的一部分导致错误累积。能够在训练的时候矫正模型的预测（避免误差进一步放大），加快模型的收敛速度。
        按一定比列，更新错误预测为标签值。

    5.8 transformer （一种特殊的 注意力机制模型）
        基于 seq2seq 架构，可以完成 NLP 典型任务（翻译，文本生成），还可以预训练迁移学习需要的模型。
        BERT 底层基石， Transformer 能够实现 GPU 分布式并行训练模，能很好捕捉长间隔的语义关联。
        对于翻译任务，文本嵌入层 （Embedding 层）产生的张量（词嵌入张量），其最后一维为词向量。

        a. transformer 架构
            输入部分包含：原文本嵌入层及其位置编码器（编码器输入），和译文文本嵌入层及其位置编码器（解码器输入）
            输出部分：解码器后的线性层和 softmax 层，得到输出概率
            编码器部分： 由 N 个编码器堆叠而成。其中单个编码器对于输入进行以下操作： 第一子层以输入部分作为输入，包含一个多头自注意力层（和规范化层，
                以及一个残差连接；第二子层以第一子层输出为输入，包含一个前馈全连接层和规范化层，以及一个残差连接。
            解码器部分： 由 N 个解码器堆叠而成。其中单个解码器对于输入进行以下操作： 第一子层以输入部分作为输入包含一个加入掩码的多头自注意力层和规
                范化层，以及一个残差连接；第二子层以第一子层输和出编码器输出为输入，包含一个多头注意力层和规范化层，以及一个残差连接；第三子层以第
                二子层输出为输入，包含一个前馈全连接层和规范化层，以及一个残差连接。

        b. 输入部分实现
            i. 文本嵌入层将文本中词汇的数字表示转变为高维空间的向量表示，利于捕捉词汇间的关系。输入 word2index 已经数字化的词序列
                from torch.autograd import Variable
                class Embeddings(nn.Module):    输入三维 batch_size，序列长度，词对应数字表示
                    def __init__(self, d_model, vocab):
                        # d_model 词嵌入维度， vocab 词表大小， padding_idx=0 对于输入序列长度补全默认，对应词嵌入中第 0 维对应全 0 输出
                        super(Embeddings, self).__init__()
                        self.lut = nn.Embedding(vocab, d_model)
                        self.d_model = d_model

                    def forward(self, x):   返回形状 (input.shape[0](batch_size), input.shape[1] 序列长度，词嵌入维度) 的词嵌入的张量
                        return self.lut(x) * math.sqrt(self.d_model)    嵌入结果经过缩放

                emb = nn.Embeddings(10,1000)
                emb_res = emb(Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))))     结果形状 (2，4,10)

            ii. 位置编码器： 将词汇位置不同可能产生不同语义的信息加入到词嵌入张量中，作为位置信息。
                class PositionalEncoding(nn.Module):
                    def __init__(self, d_model, dropout, max_len=5000):
                        # d_model: 词嵌入维度, dropout: 置0比率, max_len: 每个句子的最大长度
                        super(PositionalEncoding, self).__init__()
                        self.dropout = nn.Dropout(p=dropout)
                        pe = torch.zeros(max_len, d_model)    # 初始化一个位置编码矩阵 (max_len x d_model)
                        position = torch.arange(0, max_len).unsqueeze(1)  # 初始化一个形状 max_len x 1 绝对位置矩阵 (词在序列中索引)
                        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
                            # 位置矩阵赋值为变化矩阵 div_term （形状 1xd_model）与绝对位置相乘（变换形状），来添加位置信息。同时缩小位置
                            # 矩阵的值方便训练收敛，使用正弦余弦（求导快）确保了在不同位置对应嵌入向量发生变化，可以求梯度。
                        pe[:, 0::2] = torch.sin(position * div_term)   偶数位置
                        pe[:, 1::2] = torch.cos(position * div_term)   奇数位置
                        pe = pe.unsqueeze(0)
                        self.register_buffer('pe', pe)  # 注册之后可以在模型保存后重加载时和模型结构与参数一同被加载这个固定值.

                    def forward(self, x):
                        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)   # 一个维度裁切到实际句子长度，pe 不需要更新
                        return self.dropout(x)

                d_model, dropout, max_len = 512, 0.1, 60
                pe = PositionalEncoding(d_model, dropout, max_len)
                pe_res = pe(emb_res)

                可视化。每条（正弦/余弦）曲线（范围[-1,1]）代表一个词汇的特征在不同位置的含义，
                    plt.figure(figsize=(15, 5))
                    pe = PositionalEncoding(20, 0)
                    y = pe(Variable(torch.zeros(1, 100, 20)))
                    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())  查看特征 4-8
                    plt.legend(["dim %d"%p for p in [4,5,6,7]])

        c. 编码器部分实现
            i. 掩码张量
                张量中一般只有 0 或 1 元素， 代表位置是否被遮掩，作用再其他张量使其一些元素被遮掩。在 Transformer 中作用是遮掩文本嵌入层
                张量中当前序列步之后的一些未来的信息，此处 1 为遮掩。
                def subsequent_mask(size):
                     # size是掩码张量最后两个维度的大小（相同）
                    attn_shape = (1, size, size)
                    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  形成上三角阵数值为 1 矩阵，
                        # k=1 从主对角线向上平移1个格，开始保留数据，下方清零
                    return torch.from_numpy(1 - subsequent_mask)   反转为包含主对角线的下三角阵
                plt.imshow(subsequent_mask(20)[0])

            ii. 注意力机制
                使用第三种注意力计算规则：Attention(Q,K,V) = softmax(Q·K^T/sqrt(d_k))·V。
                query 理解为文本， key 理解为关键词， value 理解为文本对应的答案。最初 value 类似 key，训练之后 value 发生变化，更能通
                过关键词查询文本。默认 key = value, query 不同。当 query = key = value 时为自注意力机制，从文本自身提取关键词，相当于
                使用特征提取。
                def attention(query, key, value, mask=None, dropout=None):
                    # mask 掩码张量, dropout nn.Dropout层的实例化对象
                    d_k = query.size(-1)   # 最后一维，词嵌入维度
                    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)   # key最后维度是词嵌入

                    if mask is not None:
                        scores = scores.masked_fill(mask == 0, -1e9)    如果 mask 对应位置为 0，则 score 对应位置的值变为 -1e9

                    p_attn = F.softmax(scores, dim = -1)

                    if dropout is not None:
                        # 将p_attn传入dropout对象中进行'丢弃'处理
                        p_attn = dropout(p_attn)
                    return torch.matmul(p_attn, value), p_attn

                query = key = value = pe_res
                attn, p_attn = attention(query, key, value)   返回注意力张量，注意力表示

            iii. 多头注意力机制

        d. 解码器部分实现


        e. 输出部分实现

        f. 模型构建

        g. 模型测试运行




--------------------------------------------------------------------------------------------------------------
    # 词嵌入练习
    from torchtext.datasets import text_classification
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import random_split

    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='./data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = 32
    NUM_CLASS = len(train_dataset.get_labels())
    N_EPOCHS = 20

    train_len = int(len(train_dataset) * 0.8)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])  随机划分训练，验证集

    class TextSentiment(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_class):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)   sparse 代表只更新部分权重
            self.fc = nn.Linear(embed_dim, num_class)
            self.init_weights()

        def init_weights(self):
            initrange = 0.5
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()

        def forward(self, text):
            embedded = self.embedding(text, offsets)  返回 （词汇总数 ,词向量尺寸） 矩阵
            s = embedded.size(0) // batch_size        # embedded.size(0) 为数据中词汇总数
            embedded = embedded[:BATCH_SIZE * s]     获取前 BATCH_SIZE * s 行
            embedded = embedded.transpose(1,0).unsqueeze(0)   返回（1, 词向量尺寸, BATCH_SIZE * s） 矩阵，平均池化需要 3 维输入
            embedded = F.avg_pool1d(embedded, kernel_size=s)  一维平均池化作用于最后一个维度，返回 （1，词向量尺寸, BATCH_SIZE ）
            return self.fc(embedded[0].transpose(1,0))    返回 （BATCH_SIZE，词向量维度） 矩阵

    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)    步长学习率衰减优化器

    def generate_batch(batch):    batch: [(sample1, label1), (sample2, label2), ...]
        label = torch.tensor([entry[1] for entry in batch])
        text = [entry[0] for entry in batch]
        text = torch.cat(text)   对 sample 张量进行拼接
        return text, label    返回张量 [*sample1,*sample2,...], [label1,label2,...]

    def train(train_data):
        train_loss, train_acc = 0,0
        data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True. collate_fn=generate_batch)
        for i, (text,cls) in enumerate(data):
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(predictions, batch.label)
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == cls).sum().item()
        scheduler.step()
        return train_loss / len(data), train_acc / len(data)

    def valid(valid_data):
        valid_loss, valid_acc= 0, 0
        data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)
        for text, cls in data:
            with torch.no_grad():      验证阶段,模型的参数不求梯度
                output = model(text)    对 text 求 embedding 后张量
                try:
                    loss = criterion(output, cls)
                except:
                    continue
                valid_loss += loss.item()
                valid_acc += (output.argmax(1) == cls).sum().item()
        return valid_loss / len(valid_data), valid_acc / len(valid_data)


    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(sub_train_)
        valid_loss, valid_acc = valid(sub_valid_)

        secs = int(time.time() - start_time)
        mins, secs = secs / 60, secs % 60
        print('Epoch: %d' % (epoch + 1), " | time in %d minites, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    print(model.state_dict()['embedding.weight'])  词嵌入（BATCH_SIZE，词向量维度） 矩阵
    torch.save(model.state_dict(), MODEL_PATH)
    # model.load_state_dict(torch.load(MODEL_PATH))



    # RNN, LSTM, GRU  输入人名，预测国家
    import glob
    import string
    import unicodedata
    from io import open

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    data_path = "./data/name/"

    def unicodeToAscii(s):    # 去除重音标记
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn' and c in all_letters)

    def readLines(filename):  # 读取文件，返回字符串列表
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]

    category_lines, all_categories = {}, []
    for filename in glob.glob(data_path + '*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    def lineToTensor(line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][all_letters.find(letter)] = 1
        return tensor

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            # RNN输入最后一维尺寸, RNN的隐层最后一维尺寸, RNN层数
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN(input_size, hidden_size, num_layers)
            self.linear = nn.Linear(hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=-1)

        def forward(self, input, hidden):
            # input 形状是 1 x n_letter, shidden 隐层张量形状 self.num_layers x 1 x self.hidden_size
            input = input.unsqueeze(0)     #RNN要求输入维度一定是三维张量
            rr, hn = self.rnn(input, hidden)
            return self.softmax(self.linear(rr)), hn


        def initHidden(self):
            return torch.zeros(self.num_layers, 1, self.hidden_size)

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
            self.linear = nn.Linear(hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=-1)


        def forward(self, input, hidden, c):
            input = input.unsqueeze(0)
            rr, (hn, c) = self.lstm(input, (hidden, c))
            return self.softmax(self.linear(rr)), hn, c

        def initHiddenAndC(self):
            c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
            return hidden, c

    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(GRU, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers)
            self.linear = nn.Linear(hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=-1)

        def forward(self, input, hidden):
            input = input.unsqueeze(0)
            rr, hn = self.gru(input, hidden)
            return self.softmax(self.linear(rr)), hn

        def initHidden(self):
            return torch.zeros(self.num_layers, 1, self.hidden_size)

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def categoryFromOutput(output):
        top_v, top_i = output.topk(1)  # 返回值，索引
        category_i = top_i[0].item()
        return all_categories[category_i], category_i

    def randomTrainingExample():
        category = random.choice(all_categories)
        line = random.choice(category_lines[category])
        category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
        line_tensor = lineToTensor(line)
        return category, line, category_tensor, line_tensor

    def trainRNN(category_tensor, line_tensor):
        # category_tensor类别标签, line_tensor 名字训练数据
        hidden = rnn.initHidden()
        rnn.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        loss = criterion(output.squeeze(0), category_tensor)
        loss.backward()
        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)
        return output, loss.item()

    def trainLSTM(category_tensor, line_tensor):
        hidden, c = lstm.initHiddenAndC()
        lstm.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden, c = lstm(line_tensor[i], hidden, c)
        loss = criterion(output.squeeze(0), category_tensor)
        loss.backward()
        for p in lstm.parameters():
            p.data.add_(-learning_rate, p.grad.data)
        return output, loss.item()

    def trainGRU(category_tensor, line_tensor):
        hidden = gru.initHidden()
        gru.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden= gru(line_tensor[i], hidden)
        loss = criterion(output.squeeze(0), category_tensor)
        loss.backward()
        for p in gru.parameters():
            p.data.add_(-learning_rate, p.grad.data)
        return output, loss.item()

    def evaluateRNN(line_tensor):
        hidden = rnn.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        return output.squeeze(0)

    def evaluateRNN(line_tensor):
        hidden = rnn.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        return output.squeeze(0)

    def evaluateGRU(line_tensor):
        hidden = gru.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = gru(line_tensor[i], hidden)
        return output.squeeze(0)

    input_size = n_letters
    n_hidden = 128
    output_size = n_categories
    input = lineToTensor('B').squeeze(0)    # 假设一个字母B作为RNN的首次输入
    hidden = c = torch.zeros(1, 1, n_hidden)
    rnn = RNN(n_letters, n_hidden, n_categories)
    lstm = LSTM(n_letters, n_hidden, n_categories)
    gru = GRU(n_letters, n_hidden, n_categories)
    rnn_output, next_hidden = rnn(input, hidden)
    lstm_output, next_hidden, c = lstm(input, hidden, c)
    gru_output, next_hidden = gru(input, hidden)
    print(categoryFromOutput(gru_output))


    criterion = nn.NLLLoss()
    learning_rate = 0.005
    n_iters = 1000
    print_every = 50
    plot_every = 10

    def train(train_type_fn):    # train_type_fn 模型训练函数
        all_losses = []
        start = time.time()
        current_loss = 0
        for iter in range(1, n_iters + 1):
            category, line, category_tensor, line_tensor = randomTrainingExample()
            output, loss = train_type_fn(category_tensor, line_tensor)
            current_loss += loss
            if iter % print_every == 0:
                guess, guess_i = categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
        return all_losses, int(time.time() - start)

    def predict(input_line, evaluate, n_predictions=3):
        # evaluate 评估函数  n_predictions 最有可能的 top个
        with torch.no_grad():
            output = evaluate(lineToTensor(input_line))
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []
            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])

    all_losses1, period1 = train(trainRNN)
    all_losses2, period2 = train(trainLSTM)
    all_losses3, period3 = train(trainGRU)

    plt.figure(0)
    plt.plot(all_losses1, label="RNN")
    plt.plot(all_losses2, color="red", label="LSTM")
    plt.plot(all_losses3, color="orange", label="GRU")
    plt.legend(loc='upper left')

    plt.figure(1)
    x_data=["RNN", "LSTM", "GRU"]
    y_data = [period1, period2, period3]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)

    line_tensor = lineToTensor("Bai")
    rnn_output = evaluateRNN(line_tensor)

    for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]:
        predict('Dovesky', evaluate_fn)
        predict('Jackson', evaluate_fn)
        predict('Satoshi', evaluate_fn)


    # seq2seq Attention 英译法
    from io import open
    import unicodedata
    import re
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token = 0    # 起始标志
    EOS_token = 1     # 结束标志

    class Lang:
        def __init__(self, name):     # name 语言的名
            self.name = name
            self.word2index = {}  # 初始化词汇对应自然数值的字典
            self.index2word = {0: "SOS", 1: "EOS"}
            self.n_words = 2

        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)      # 在.!?前加一个空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

data_path = '../Downloads/data/eng-fra.txt'

def readLangs(lang1, lang2):   # lang1是源语言的名字, 参数lang2是目标语言的名字
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


MAX_LENGTH = 10   # 设置组成句子中单词或标点的最多个数
eng_prefixes = (   # 选择带有指定前缀的语言特征数据作为训练数据
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):      # p代表输入的语言对
    return len(p[0].split(' ')) < MAX_LENGTH and p[0].startswith(eng_prefixes) and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra')

def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # input_size 解码器的输入尺寸即源语言的词表大小，hidden_size代表GRU的隐层节点数, 也代表词嵌入维度, 同时又是GRU的输入尺寸
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(dropout_p)
    def forward(self, input, hidden):
        # input代表源语言的Embedding层输入张量 hidden代表编码器层gru的初始隐层张量
        output = self.embedding(input).view(1, 1, -1)     #尺寸是[1, embedding]
        # output  = self.dropout(output)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        # hidden_size 解码器中GRU隐层节点数， output_size 解码器输出尺寸, 目标语言的词表大小
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        # input 目标语言的 Embedding 层输入张量， hidden 解码器 GRU 隐层张量
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 模式 a
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # hidden_size 解码器中GRU的输入尺寸，隐层节点数   output_size 解码器的输出尺寸, 目标语言的词表大小
           # dropout_p 置零比率, max_length代表句子的最大长度
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        # 源数据输入张量, 初始的隐层张量, 以及解码器的输出张量
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 进行attention的权重计算, 哦我们呢使用第一种计算方式：
        attn_weights = F.softmax(self.attn(torch.cat((hidden[0],encoder_outputs), 1)), dim=1)
        # 然后进行第一步的后半部分, 将得到的权重矩阵与V做矩阵乘法计算, 当二者都是三维张量且第一维代表为batch条数时, 则做bmm运算
        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 之后进行第二步, 通过取[0]是用来降维, 根据第一步采用的计算方法, 需要将Q与第一步的计算结果再进行拼接
        output = torch.cat((embedded[0], context_vector[0]), 1)
        # 最后是第三步, 使用线性层作用在第三步的结果上做一个线性变换并扩展维度，得到输出
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 模式 b
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.output_size = output_size  # 输出层大小，实际上为目标语言的词典大小
        self.embedding = nn.Embedding(
            self.output_size, self.hidden_size)  # 词向量层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)  # GRU 层
        self.out = nn.Linear(self.hidden_size*2, self.output_size)  # 输出层

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = F.relu(embedded)  # 激活函数 Relu
        output, hidden = self.gru(embedded, hidden)

        attn_weights = F.softmax(torch.bmm(encoder_outputs.unsqueeze(
            0), hidden.view(1, self.hidden_size, -1)), dim=1)
        weighted_context = torch.matmul(
            attn_weights.squeeze(2), encoder_outputs)
        output = torch.cat([output.squeeze(0), weighted_context], dim=1)
        output = F.softmax(self.out(output), dim=1)
        return output, hidden, attn_weights


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # input_tensor：源语言输入张量，target_tensor：目标语言输入张量，criterion：损失函数计算方法，max_length：句子的最大长度
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)   # 初始化解码器的第一个输入，即起始符
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            # 并强制将下一次的解码器输入设置为‘正确的答案’
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            loss += criterion(decoder_output, target_tensor[di])
            if topi.squeeze().item() == EOS_token:
                break
            decoder_input = topi.squeeze().detach()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    # n_iters: 总迭代步数, print_every:打印日志间隔, plot_every:绘制损失曲线间隔
    start = time.time()
    plot_losses = []
    print_loss_total, plot_loss_total = 0, 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    for iter in range(1, n_iters + 1):
        training_pair = tensorsFromPair(random.choice(pairs))
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    plt.figure()
    plt.plot(plot_losses)
    plt.savefig("./s2s_loss.png")

    def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
            return decoded_words, decoder_attentions[:di + 1]

    def evaluateRandomly(encoder, decoder, n=6):
        for i in range(n):
            pair = random.choice(pairs)
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print(pair, '<', output_sentence)

    hidden_size = 25
    input_size = 20
    output_size = 10
    teacher_forcing_ratio = 0.5
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    n_iters = 75000
    print_every = 5000
    trainIters(encoder1, attn_decoder1, n_iters, print_every=print_every)
    sentence = "we re both teachers ."
    output_words, attentions = evaluate(encoder1, attn_decoder1, sentence)
    print(output_words)
    plt.matshow(attentions.numpy())
    plt.savefig("./s2s_attn.png")



'''
import torch.nn as nn
nn.Embedding

import jieba
import sklearn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
import pandas as pd
import matplotlib.pyplot as plt
plt.hist()
df = pd.DataFrame([[1,2],[3,4]], columns=['a','b'])
s =pd.Series([1])
s.des
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
sc = CountVectorizer
x_reg = sc.t
import matplotlib.pyplot as plt
pd.crosstab()
v = sklearn.feature_extraction.DictVectorizer(sparse=True)
v.fit_transform
from sklearn.ensemble import RandomForestClassifier
est = RandomForestClassifier
x=jieba.cut("我爱北京天安门")
from sklearn.linear_model import SGDRegressor
import torchvision.transforms as transforms

import torch
x = torch.rand(5,3)
import torchvision
torchvision.datasets.CIFAR10
torchvision.utils.make_grid(images)
import torch.nn as nn
nn.LSTM