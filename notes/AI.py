'''
数据库 http://archive.ics.uci.edu/ml/
Microsoft Azure Machine Learning studio: 实验左侧所有操作功能，中间流程图连接各个操作，右侧具体当前操作参数
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
            深度学习是机器学习的一个方法发展而来
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
        e. 特征工程
            i. 定义：使用专业背景知识和技巧处理数据，使得特征能在机器学习算法上发挥更好的作用的过程
            数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。
            ii. 包含内容：
                特征提取： 将数据转化为可用于机器学习的数字特征
                特征预处理： 通过转换函数将特征数据转换成更加适合算法模型的特征数据过程
                特征降维： 降低随机变量（特征）个数，简化模型的过程
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
    2.2 ** Jupyter Notebook 使用
        a. 简介
            开源的科学计算平台，类别ipython，可以运行代码，做笔记，画图表。文件后缀 .ipynb
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

    2.3 Matplotlib
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
            plt.hist(x, bins=None)
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

    2.5 Numpy
        a. 定义
            numpy 是一个开源的用于快速处理多维数组的库
            存储对象是 ndarry， 使用 np.array([], dtype=np.int16) 多维数组创建
            ndarry 数据类型： bool, int, float, string_, unit (正整数)，complex，object_(python对象)，unicode_

        b. 优势 （速度快，有很多数组运算的 api）
            python 原生数组为了满足存不同类型数据，需通过数组存储的内存地址一个个查询对应的数据，而 numpy 直接将同一数据类型的数组
                （不同类型会从 int -> float -> string）存在连续的地址中
            numpy 支持并行化（向量化）运算
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
                    b = np.array(a) # 深拷贝, 改 a 值不变
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

    2.6 pandas
        a. 定义
            开源的数据挖掘库，用于数据探索。以 numpy 为基础（高性能），基于 matplotlib（画图），有独特的数据结构
        b. 优势
            便捷的数据处理能力，读取文件，封装了 matplotlib,numpy
        c.Series
            s.unique()
            自定义运算
                s.apply(lambda x:2*x)      Series can't use max() for lambda
                s.transform(lambda x:x.max()-x)
                s.map(lambda x:2*x)
        d. DataFrame
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
            html, clipboard, excel, parquet, pickle, sql, gbq
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
            df['day'] = pd.to_datetime(data.index).weekday
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
        3.1 scikit-learn
        文档多，且规范。包含的算法多，实现容易。但是不支持神经网络和深度学习。
        包含分类，回归，聚类，降维，模型选择和预处理。
        d = sklearn.datasets.load_*()     ex. load_iris 获取库中本地小规模数据集
        d = sklearn.datasets.fetch_*(data_home=None, subset='train')    ex.fetch_20newsgroups 获取属于库在网络中较大规模数据集，
            默认下载到 ~/scikit_learn_data 目录下, subset 可选 train(默认), test, all, 中途超时，尝试多次下载
            返回数据 Bunch 类似字典，data(特征数组 二维 numpy ndarray)，target(标签数组 一维 ndarray)，feature_names(特征名 list)，
            target_names(ndarray 目标值所有可能取值), DESCR(字符串数据描述)   使用 d.data 或 d["data"] 获取
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.2, random_state=None)
            将数据集分为训练集和测试集, 测试集占 20%， 传入数据集类型为 list, numpy ndarry 和 pandas.DataFrame
        estimator.fit(x,y)  训练数据 x 必须 2 维， 标签值 y 必须 1 维。 训练
        estimator.predict(x)       预测
        estimator.score(x,y)       评估

        3.2 K-近邻（KNN）算法
            如果一个样本在特征空间中的k个最相似(即特征空间中最邻近，通常使用欧式距离)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=6)        # 默认参考 5 个邻居
            k 值选择： 选择奇数，过小容易收到异常点影响（过拟合），过大受到样本均衡的问题（欠拟合）。
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
            kd 树, ball tree: 把距离信息保存在二叉排序树中，使得搜索时间复杂度降低到 O(DNlog(N))。基本原理是：如果 A 和 B 距离很远， B
                和 C 距离很近，那么 A 和 C 距离也很远。
                构造 kd 树，选取中间的节点（所在方差最大的维度两侧节点数相同或差1），递归将数据空间通过此节点切分成两组（下一轮选取维度与这轮
                不同）对应左子树和右子树，直到叶子节点（空间中无数据点）。查找（判断新数据点位置）类似二分查找，依次和树的当前节点的切分维度的值
                对比，小则进入左子树，大则进入右子树，直到叶子节点，经过的节点记入栈中。计算从需判断点 g 到该叶子点的距离 r，再以 g 为圆心, r
                为半径判断圆是否与之前栈节点 a 的切分超平面相交。如果不相交则 a 出栈。如果相交，记录 a 到 g 的距离 r'，如果比 r' < r，则更
                新 r = r'，a 出栈，同时将 a 另一侧的子树根节点入栈。直到栈中没有节点。
        e. 特征工程
            i. 定义：使用专业背景知识和技巧处理数据，使得特征能在机器学习算法上发挥更好的作用的过程
            数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。
            ii. 包含内容：
                特征提取： 将数据转化为可用于机器学习的数字特征
                特征预处理： 通过转换函数将特征数据转换成更加适合算法模型的特征数据过程
                特征降维： 降低随机变量（特征）个数，简化模型的过程
            归一化
                通过对原始数据进行变换把数据映射到[0,1]  x' = (x- min(x)) / (max(x) - min(x))  x" = x'*(hx-lx)+lx
                    hx, lx: 为转换后的最大值最小值，默认为 1,0
                sc = sklearn.preprocessing.MinMaxScaler()
                x_reg = sc.fit_transform(x)     x 可以是列表，ndarray，DataFrame 等
                或[-1,1]之间，通过这个变换使数据特征的均值变为0，标准差变为1
'''
import sklearn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
import pandas as pd
df = pd.DataFrame([[1,2],[3,4]], columns=['a','b'])
s =pd.Series([1])
s.des
import seaborn as sns
sklearn.model_selection.train_test_split
sc = sklearn.preprocessing.MinMaxScaler()
x_reg = sc.fit_transform(x)
import matplotlib.pyplot as plt
pd.crosstab()