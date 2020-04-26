## Paillier Operator on GPU 接口测试

### 测试环境

- 密钥长度：1024 bit
- 计算资源：Nvidia Tesla V100
- cuda 10.0

### 测试列表

- encrypt_and_obfuscate
- decryptdecode
- add
- scala_mul
- mul
- vdot
- matmul
- transe
- mean
- hstack

### 测试

encrypt_and_obfuscate

|    | 参数名  | 参数类型                                | 注释       |
|----|------|-------------------------------------|----------|
| 参数 | data | numpy\.ndarray\(dtype=np\.float64\) | 也可以是列表   |
|    | pub  | PaillierPublicKey                   |          |
|    | obf  | bool                                | 是否进行混淆操作 |


测试效果

balabala

decryptdecode

|    | 参数名  | 参数类型                                              | 注释       |
|----|------|---------------------------------------------------|----------|
| 参数 | data | numpy\.ndarray\(dtype=PaillierEncryptedNumber\)\) | 也可以是列表   |
|    | pub  | PaillierPublicKey                                 |          |
|    | priv | PaillierPrivateKey                                |  |

测试效果

balabala

add

|    | 参数名  | 参数类型                                              | 注释       |
|----|------|---------------------------------------------------|----------|
| 参数 | x | numpy\.ndarray\(dtype=PaillierEncryptedNumber\)\) | 也可以是列表   |
|    | y | numpy\.ndarray\(dtype=PaillierEncryptedNumber\)\) | 也可以是列表 |
|    | pub  | PaillierPublicKey                                 |          |

测试效果

balabala


scala_mul

|    | 参数名  | 参数类型                                              | 注释       |
|----|------|---------------------------------------------------|----------|
| 参数 | x | numpy\.ndarray\(dtype=PaillierEncryptedNumber\)\) | 也可以是列表   |
|    | s | float |  |
|    | pub  | PaillierPublicKey                                 |          |


mul

|    | 参数名  | 参数类型                                              | 注释       |
|----|------|---------------------------------------------------|----------|
| 参数 | x | numpy\.ndarray\(dtype=PaillierEncryptedNumber\)\) | 也可以是列表   |
|    | y | numpy\.ndarray\(dtype=float64\)\) | 也可以是列表 |
|    | pub  | PaillierPublicKey                                 |          |

测试效果

balabala

vdot

|    | 参数名  | 参数类型                                              | 注释       |
|----|------|---------------------------------------------------|----------|
| 参数 | x | numpy\.ndarray\(dtype=PaillierEncryptedNumber\)\) | 也可以是列表   |
|    | y | numpy\.ndarray\(dtype=float64\)\) | 也可以是列表 |
|    | pub  | PaillierPublicKey                                 |          |


测试效果

balabala

matmul

|    | 参数名  | 参数类型                                              | 注释       |
|----|------|---------------------------------------------------|----------|
| 参数 | x | numpy\.ndarray\(dtype=PaillierEncryptedNumber\)\) | 也可以是列表   |
|    | y | numpy\.ndarray\(dtype=float64\)\) | 也可以是列表 |
|    | pub  | PaillierPublicKey                                 |          |


测试效果

balabala