main
``` sudo

FOR day_idx in range days:
	initalize_a_day()
	WHILE true:
		SWITCH:
			add_attr()
			add_meal()
			{
			end_with_hotel_or_train()
			break;
			}
```

plan_core:
``` sudo
class plancore:

attributes:
	plan: type=object
	'''
	e.g.:
	plan = {
		action1 ： ｛
			type: transcity_travel:train,
			start_time: d1_13h,
			end_time: d1_23h,
			plot: 3 
			...
		},
		action2: {...}
	}
	'''

	
```
plan_planner
``` sudo
class plan:

atttibutes:
	history：[]plan_core
	future: []plan_core

method:
add_node()
	commit_node()
	undo_node()

```

raw:
``` sudo
算法 DFS_POI(query, poi_plan, plan, current_time, current_position, current_day=0)
输入: 
    query: 用户查询参数
    poi_plan: POI规划信息
    plan: 当前行程计划
    current_time: 当前时间
    current_position: 当前位置
    current_day: 当前天数
    
输出: (success, plan) - 成功标志和最终计划

开始:
    1. 搜索节点数加1
    2. 检查超时条件:
        - 如果超过最大搜索时间，抛出超时异常
    
    3. 约束检查:
        - 检查是否太晚无法返回酒店或交通点，如果是则回溯
        - 检查预算是否超支，如果是则回溯
    
    4. 处理跨城交通（第一天）:
        - 如果是第一天且当前时间为空
        - 添加上程交通到计划
        - 递归调用DFS_POI
    
    5. 处理早餐:
        - 如果当前时间为"00:00"
        - 选择并添加早餐活动
        - 递归调用DFS_POI
    
    6. 生成候选活动类型:
        - 根据当前状态生成: attraction, lunch, dinner, hotel, back-transport等
    
    7. 循环处理每个候选类型:
        While 候选类型列表不为空:
            a. 选择下一个POI类型
            b. 根据类型分别处理:
            
            b1. 返程交通处理:
                - 排名市内交通方式
                - 尝试每种交通方式
                - 验证约束条件
                - 如果成功则返回最终计划
            
            b2. 酒店住宿处理:
                - 选择酒店并计算房间需求
                - 排名市内交通方式
                - 添加住宿活动
                - 递归调用DFS_POI（下一天）
            
            b3. 餐饮景点处理:
                - 餐厅: 排名并选择餐厅，考虑约束条件
                - 景点: 排名并选择景点，检查开放时间
                - 添加相应活动到计划
                - 递归调用DFS_POI
            
            c. 如果当前类型所有尝试失败，从候选列表移除该类型
    
    8. 返回失败结果

算法 GENERATE_PLAN_WITH_SEARCH(query)
开始:
    1. 收集跨城交通信息（火车、飞机）
    2. 初始化搜索参数
    3. 对每种上程交通方式排序:
        For 每种上程交通方式:
            For 每种返程交通方式:
                If 行程天数>1:
                    For 每种酒店选择:
                        a. 验证房间需求约束
                        b. 计算交通和住宿总成本
                        c. 调用DFS_POI进行详细规划
                Else:
                    a. 验证时间约束
                    b. 调用DFS_POI进行一日游规划
    
    4. 返回最终结果或错误信息



```