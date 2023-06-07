---
layout: single
title:  "깃허브 Git Push 오류 해결 (unable to access)"
categories: "Git-and-Github"
tags: [git, github, error]
author_profile: false
toc: true
toc_sticky: true
toc_label: 목차
---

## 1. 깃허브 오류 발생 (unable to access)

![230607-gh](https://github.com/SukyungJang/baekjoon/assets/133842344/ae5762df-d1d5-49e9-8c03-7a1781af7b71)

깃허브 계정을 변경하고 저장소를 로컬로 옮기니 해당 오류가 발생했습니다. 

## 2. 이름과 이메일을 원하는 것으로 변경

![230607-gh1](https://github.com/SukyungJang/baekjoon/assets/133842344/4a089767-85a7-4327-ad3a-040b5f441dc0)

먼저 저는 이름을 저의 이름으로 변경해주고 이메일 역시 제가 사용하려는 이메일로 변경하였습니다.

## 3. 로그인 정보 삭제

![230607-gh2](https://github.com/SukyungJang/baekjoon/assets/133842344/9474d88b-c237-4e3d-8ee3-df6b6d7ed7b4)

검색창에서 **제어판을 검색한 후 사용자 계정 → 자격 증명 관리자**로 들어갑니다.

그 다음 Windows 자격 증명을 누르면 위의 사진처럼 내용이 나옵니다.

그리고 일반 자격 증명을 보게 되면 깃허브 관련 url이 보이고 있습니다.

저 url을 누르게 되면 정보와 편집 또는 제거할 수 있는데 전부 제거해줍니다.

## 4. git push 후 로그인

![230607-gh3](https://github.com/SukyungJang/baekjoon/assets/133842344/d00d3aa3-0fe8-4916-a895-c82bf56f2efc)

![230607-gh4](https://github.com/SukyungJang/baekjoon/assets/133842344/cdc7b32f-84a7-4122-a77e-b9e74d0bfce4)

다시 Git Bash로 돌아가서 git push를 하게 되면 위의 사진처럼 로그인하는 창이 나옵니다.

로그인을 하면 정상적으로 push가 가능합니다.