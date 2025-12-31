# 在虚拟机（VM）中从零搭建 SQL Server（Docker）开发数据库

本文档记录了**在一台开发用虚拟机中，从零开始搭建 SQL Server（Docker）并成功导入中文 CSV 数据**的完整流程。  
目标是：**开机即可用、可重复、可维护、不踩编码坑**。

---

## 一、环境假设

- 宿主机：Windows / macOS（仅用于运行 VM）
- 虚拟机系统：Ubuntu 22.04 LTS
- Docker：已安装（官方方式）
- Docker Compose：v2
- 数据库：SQL Server 2022 (Linux, Developer Edition)
- 使用场景：开发环境（非生产）

---

## 二、安装与基础检查

### 1. 检查 Docker 是否正常

```bash
docker --version
docker compose version
```

### 2. 确认 Docker 开机自启（推荐）

```bash
systemctl is-enabled docker
systemctl status docker --no-pager
```

---

## 三、目录结构规划（推荐）

```text
/opt/wenshu/
├── docker-compose.yml
├── .env
├── import/
│   ├── FACT_TASK_ASSIGN.csv
│   └── FACT_TASK_ASSIGN_utf16bom.csv
├── init_fact_tasks.sql
└── db/
    └── mssql/
```

---

## 四、准备 docker-compose.yml

```yaml
services:
  mssql:
    image: mcr.microsoft.com/mssql/server:2022-latest
    container_name: wenshu-mssql
    restart: unless-stopped
    environment:
      ACCEPT_EULA: "Y"
      MSSQL_PID: "Developer"
      SA_PASSWORD: ${SA_PASSWORD}
      TZ: Asia/Shanghai
    ports:
      - "127.0.0.1:1433:1433"
    volumes:
      - ./db/mssql:/var/opt/mssql
      - ./import:/import
```

---

## 五、准备 .env 文件

```env
SA_PASSWORD=StrongPassword@2025
```

---

## 六、启动数据库容器（只需一次）

```bash
cd /opt/wenshu
docker compose up -d
docker ps
```

---

## 七、处理中文 CSV（关键步骤）

### 转换为 UTF-16（带 BOM）

```bash
iconv -f UTF-8 -t UTF-16 FACT_TASK_ASSIGN.csv > FACT_TASK_ASSIGN_utf16bom.csv
```

验证：

```bash
xxd -g 1 -l 4 FACT_TASK_ASSIGN_utf16bom.csv
```

---

## 八、执行初始化 SQL

```bash
source /opt/wenshu/.env

docker run --rm -it   --network container:wenshu-mssql   -v /opt/wenshu/import:/import   mcr.microsoft.com/mssql-tools   /opt/mssql-tools/bin/sqlcmd     -b -V16     -S localhost     -U sa     -P "$SA_PASSWORD"     -C     -i /import/init_fact_tasks.sql
```

---

## 九、验证

```sql
USE fact_tasks;
SELECT COUNT(*) FROM dbo.FACT_TASK_ASSIGN;
SELECT TOP 10 activity_name FROM dbo.FACT_TASK_ASSIGN;
```

---

## 十、设计说明

- VM 中数据库容器长期运行
- `restart: unless-stopped` 实现开机即用
- 数据导入不需要重启容器

---

## 十一、经验总结

> Linux SQL Server + 中文 CSV  
> **UTF-16 + DATAFILETYPE='widechar' 是唯一稳定方案**

---

完
