'''
数据库 http://archive.ics.uci.edu/ml/
Microsoft Azure Machine Learning studio: 实验左侧所有操作功能，中间流程图连接各个操作，右侧具体当前操作参数
kaggle， 天池  机器学习比赛
ssl 883 error 解决方法:
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context

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
'''

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

