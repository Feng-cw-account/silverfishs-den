#history #computer-science

| stages                   | pros                                                      | cons                                        | thoughts                                 |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------- | ---------------------------------------- |
| direct allocating        | direct and easy                                           | not safe, usage rate is low,addr not static | direct                                   |
| static partitioning      | supports multiprogramming                                 | severe **internal fragmentation**           | pre-divide memo before task setup        |
| **Dynamic Partitioning** | \* virtual addr,safe memory for Tasks,usage rate is high. | need extra computation                      | Dynamic allocating based on actual need. |
| Garbage collecting       | pass                                                      |                                             |                                          |

the most important stage is [[dynamic-partitioning]]

## links
[[dynamic-partitioning]]