# TC720 OpenWrt / WireGuard / Portpass / Camera Integration Standard

> 작성 목적: 본 문서는 현재 채팅에서 수행한 TC720 OpenWrt LTE 라우터, WireGuard, 제조사 포트패스, HMI/PLC REST API, Hikvision 카메라, 향후 VPN 1:1 매핑 설계를 빠짐없이 통합 기록하기 위한 기준 문서이다.  
> 저장 위치: `WireGuard_WebRTC_Test/Docs/TC720_OpenWrt_WireGuard_Portpass_Camera_Standard.md`  
> 주의: 이 문서는 AWS 관련 문서가 아니라 현장 LTE 라우터/WireGuard/WebRTC/카메라 연동 실험 및 표준화 문서이다.

---

## 1. 최종 요약

현재까지 확인된 핵심 결론은 다음과 같다.

1. TC720 라우터는 OpenWrt 19.07 기반이며 기존 `uhttpd` 웹서버가 `8080`에서 동작한다.
2. 별도 API 서버를 새로 띄운 것이 아니라, 기존 `uhttpd`의 CGI 기능을 이용해 `/www/cgi-bin/router-info`를 추가했다.
3. PLC/HMI는 RESTful API Client 기능으로 라우터의 CGI URL을 GET 호출하면 된다.
4. 대표 시스템의 중앙 WireGuard 기준 인터페이스는 `wg2`로 표준화한다.
5. 제조사 포트패스용 WireGuard 인터페이스 `wg`는 변경하지 않는다.
6. `wg3`는 AWS 테스트용이었으나 현재 AWS 인스턴스가 삭제되어 운영 대상에서 제외한다.
7. HMI/PLC가 사용할 API 응답 키는 `vpn_ip` 단일 키로 표준화한다.
8. 제조사 무료 포트패스 `10521`은 한 번에 한 대상만 전달 가능하다.
9. 포트패스를 라우터 관리화면, 카메라 Web, 카메라 RTSP로 임시 전환할 수 있다.
10. 장기 운영은 제조사 포트패스보다 대표 시스템 WireGuard(`wg2`) 기반의 `192.168.0.X <-> 10.<group>.<router>.X` 1:1 VPN 매핑이 적합하다.

---

## 2. 대상 장비 및 리소스

### 2.1 라우터 기본 정보

| 항목 | 값 |
|---|---|
| 장비 | TELADIN TC720 LTE Router |
| OS | OpenWrt 19.07-SNAPSHOT |
| Revision | `r11436-1da2e82` |
| Kernel | Linux 4.14.275 |
| Target | `ramips/mt76x8` |
| Architecture | `mipsel_24kc` |
| SoC | MediaTek MT7628AN |
| CPU | MIPS 24KEc |
| BusyBox | v1.30.1 |

### 2.2 리소스 판단

| 항목 | 확인값 | 판단 |
|---|---:|---|
| RAM total | 약 59 MB | 저사양 |
| RAM available | 약 15 MB | 단순 CGI 가능 |
| overlay free | 약 1.6 MB | 작은 shell script 저장 가능 |
| Python | 없음 | Python 서버 부적합 |
| PHP | 없음 | PHP 방식 부적합 |
| curl | `/usr/bin/curl` | 사용 가능 |
| wget | `/usr/bin/wget` | 사용 가능 |
| nc | `/usr/bin/nc` | BusyBox 방식, `-vz` 미지원 |
| uhttpd | `/usr/sbin/uhttpd` | 사용 가능 |
| iptables | `/usr/sbin/iptables` | 사용 가능 |
| tcpdump | `/usr/sbin/tcpdump` | 사용 가능 |
| wg | `/usr/bin/wg` | 사용 가능 |

판단:

- 상시 Node/Python/Go 서버를 라우터에 올리는 방식은 부적합하다.
- 기존 `uhttpd` + shell CGI가 가장 가볍고 안전하다.
- 설정 변경은 최소화하고, 파일 1~2개 추가 및 UCI firewall redirect 변경 수준으로 제한한다.

---

## 3. 네트워크 인터페이스 정리

### 3.1 LAN 인터페이스

| 항목 | 값 |
|---|---|
| LAN bridge | `br-lan` |
| 초기 LAN IP | `192.168.1.1/24` |
| 실험 LAN IP | `192.168.0.1/24` |
| 현재 복구 확인 LAN IP | `192.168.1.1/24` |
| 목표 표준 후보 | `192.168.0.1/24` |

변경 이력:

1. 초기 상태: `192.168.1.1`
2. 현장 장비 표준화를 위해 `192.168.0.1`로 변경
3. Hikvision 카메라 기본 IP 및 제조사 포트패스 복구 테스트를 위해 다시 `192.168.1.1`로 복귀

### 3.2 WAN/LTE 계열

| 인터페이스 | 예시 IP | 설명 |
|---|---:|---|
| `eth1` | `10.41.205.220/24` | LTE/RNDIS 계열 WAN으로 추정 |
| default route | `10.41.205.221` | 외부 인터넷 기본 경로 |

### 3.3 WireGuard 인터페이스

| 인터페이스 | 예시 IP | 용도 | 운영 판단 |
|---|---:|---|---|
| `wg` | `172.16.1.127/32` | 제조사 포트패스용 | 변경 금지 |
| `wg2` | `10.77.0.2/32` | GCP 중앙 WireGuard 서버 연결 | 운영 기준 |
| `wg3` | `10.88.0.2/32` | AWS 테스트용, 인스턴스 삭제됨 | 운영 제외 |

---

## 4. WireGuard 인터페이스 표준

대량 세팅 기준은 다음으로 고정한다.

```text
wg  = 제조사 포트패스 전용. 제조사 기능 유지용. 변경 금지.
wg2 = 대표 시스템 중앙 WireGuard VPN. PLC/HMI/API/카메라 원격 접근 기준.
wg3 = 폐기 또는 테스트용. 운영 사용 금지.
```

`wg2` 상태 확인 명령:

```sh
wg show wg2
wg show wg2 latest-handshakes
ip -4 addr show wg2
```

`wg` 제조사 포트패스 상태 확인 명령:

```sh
wg show wg
ip addr show wg
```

제조사 `wg` 확인 예:

```text
interface: wg
  listening port: 42702
peer: ...
  endpoint: 112.220.220.186:51821
  allowed ips: 172.16.0.0/16
  latest handshake: 정상
  persistent keepalive: every 25 seconds
```

---

## 5. PLC/HMI용 라우터 VPN 조회 API

### 5.1 구현 방식

구현은 RESTful API 서버 프레임워크가 아니다. 기존 OpenWrt 웹서버 `uhttpd`가 HTTP 요청을 받고, CGI shell script를 실행하는 구조이다.

```text
PLC/HMI/curl
  -> HTTP GET /cgi-bin/router-info
  -> uhttpd
  -> /www/cgi-bin/router-info 실행
  -> wg2 IP 및 handshake 검사
  -> JSON 응답
```

### 5.2 현재 URL과 목표 URL

현재 `192.168.1.1` 기준:

```text
GET http://192.168.1.1:8080/cgi-bin/router-info
```

목표 `192.168.0.1` 기준:

```text
GET http://192.168.0.1:8080/cgi-bin/router-info
```

외부 제조사 포트패스를 통과해서 상태 API를 보는 경우:

```text
GET http://112.220.220.186:10521/cgi-bin/router-info
```

단, `10521`이 라우터 `192.168.1.1:8080`으로 포워딩되어 있을 때만 가능하다.

### 5.3 HTTP 상태 규칙

| 상황 | HTTP Status | 의미 | HMI 처리 |
|---|---:|---|---|
| 정상 | `200` | VPN 준비 완료 | `vpn_ip` 사용 |
| VPN IP 없음 | `503` | 인터페이스 IP 없음 | 연결 대기 |
| 기본 라우트 없음 | `503` | LTE/인터넷 준비 안 됨 | 인터넷 대기 |
| handshake 없음 | `503` | WireGuard 미연결 | VPN 대기 |
| handshake 오래됨 | `503` | 터널 stale | VPN 대기 |
| 요청 실패 | 없음 | 라우터/LAN 접근 실패 | 10초 후 재시도 |

권장값:

```text
HMI timeout: 3초
HMI 재시도: 10초
MAX_HANDSHAKE_AGE: 180초
PersistentKeepalive: 25초 권장
```

### 5.4 응답 포맷

대표님 결정에 따라 `vpn_ip` 단일 키만 사용한다. `vpn-ip`는 사용하지 않는다.

정상 응답:

```json
{
  "ok": true,
  "status": "ready",
  "iface": "wg2",
  "vpn_ip": "10.77.0.2"
}
```

대기/실패 응답:

```json
{
  "ok": false,
  "status": "waiting",
  "reason": "wireguard_handshake_stale",
  "iface": "wg2",
  "vpn_ip": ""
}
```

---

## 6. 라우터 API 설정 파일

파일:

```text
/etc/router-vpn-api.conf
```

내용:

```sh
CENTRAL_IFACE="wg2"
MAX_HANDSHAKE_AGE="180"
```

생성 명령:

```sh
cat > /etc/router-vpn-api.conf <<'SH'
CENTRAL_IFACE="wg2"
MAX_HANDSHAKE_AGE="180"
SH
chmod 600 /etc/router-vpn-api.conf
```

---

## 7. 운영용 `/www/cgi-bin/router-info`

> 이 버전은 `vpn_ip` 단일 키만 반환하는 최종 표준 후보이다.

```sh
cat > /www/cgi-bin/router-info <<'SH'
#!/bin/sh

CONFIG="/etc/router-vpn-api.conf"
CENTRAL_IFACE="wg2"
MAX_HANDSHAKE_AGE="180"
[ -f "$CONFIG" ] && . "$CONFIG"

json_escape() {
  echo "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

send_json() {
  STATUS_LINE="$1"
  BODY="$2"
  [ -n "$STATUS_LINE" ] && echo "Status: $STATUS_LINE"
  echo "Content-Type: application/json"
  echo "Cache-Control: no-store"
  echo ""
  echo "$BODY"
}

VPN_IP="$(ip -4 addr show "$CENTRAL_IFACE" 2>/dev/null | grep -oE 'inet [0-9.]+' | awk '{print $2}')"

if [ -z "$VPN_IP" ]; then
  send_json "503 Service Unavailable" "{\"ok\":false,\"status\":\"waiting\",\"reason\":\"vpn_ip_not_found\",\"iface\":\"$(json_escape "$CENTRAL_IFACE")\",\"vpn_ip\":\"\"}"
  exit 0
fi

DEFAULT_ROUTE="$(ip route 2>/dev/null | grep '^default ' | head -n 1)"
if [ -z "$DEFAULT_ROUTE" ]; then
  send_json "503 Service Unavailable" "{\"ok\":false,\"status\":\"waiting\",\"reason\":\"default_route_not_found\",\"iface\":\"$(json_escape "$CENTRAL_IFACE")\",\"vpn_ip\":\"\"}"
  exit 0
fi

WG_BIN="$(which wg 2>/dev/null)"
NOW="$(date +%s 2>/dev/null)"
if [ -z "$WG_BIN" ] || [ -z "$NOW" ]; then
  send_json "" "{\"ok\":true,\"status\":\"ready_ip_only\",\"reason\":\"wg_tool_or_time_unavailable\",\"iface\":\"$(json_escape "$CENTRAL_IFACE")\",\"vpn_ip\":\"$VPN_IP\"}"
  exit 0
fi

LATEST="$($WG_BIN show "$CENTRAL_IFACE" latest-handshakes 2>/dev/null | awk '{print $2}' | sort -nr | head -n 1)"
if [ -z "$LATEST" ] || [ "$LATEST" = "0" ]; then
  send_json "503 Service Unavailable" "{\"ok\":false,\"status\":\"waiting\",\"reason\":\"wireguard_handshake_not_yet\",\"iface\":\"$(json_escape "$CENTRAL_IFACE")\",\"vpn_ip\":\"\"}"
  exit 0
fi

AGE=$((NOW - LATEST))
if [ "$AGE" -gt "$MAX_HANDSHAKE_AGE" ]; then
  send_json "503 Service Unavailable" "{\"ok\":false,\"status\":\"waiting\",\"reason\":\"wireguard_handshake_stale\",\"iface\":\"$(json_escape "$CENTRAL_IFACE")\",\"vpn_ip\":\"\"}"
  exit 0
fi

send_json "" "{\"ok\":true,\"status\":\"ready\",\"iface\":\"$(json_escape "$CENTRAL_IFACE")\",\"vpn_ip\":\"$VPN_IP\"}"
SH
chmod +x /www/cgi-bin/router-info
```

테스트:

```sh
curl -i -m 3 http://127.0.0.1:8080/cgi-bin/router-info
curl -i -m 3 http://192.168.1.1:8080/cgi-bin/router-info
```

---

## 8. `wg3` 실패 테스트 API

나중에 지우기 쉽게 테스트 전용 이름을 사용한다.

파일:

```text
/www/cgi-bin/router-info-wg3-test
```

URL:

```text
GET http://192.168.1.1:8080/cgi-bin/router-info-wg3-test
```

목적:

- 삭제된 AWS 인스턴스용 `wg3`가 handshake 실패 또는 stale 상태로 반환되는지 HMI 실패 처리 예시를 검증한다.
- 운영에서는 사용하지 않는다.

삭제:

```sh
rm -f /www/cgi-bin/router-info-wg3-test
```

---

## 9. HMI JavaScript 연동 기준

HMI에서는 `statusCode === 200`일 때만 `vpn_ip`를 신뢰한다.

```js
function checkRouterVpn() {
  try {
    driver.setStringData(hmi.message, 100, 'VPN 연결 확인 중');

    request.get({
      url: "http://192.168.1.1:8080/cgi-bin/router-info",
      header: { "Accept": "application/json" },
      timeout: 3000
    }, (error, response, body) => {
      if (error || !response) {
        driver.setData(hmi.uiStatus, 0);
        driver.setStringData(hmi.message, 100, '라우터 응답 없음. 10초 후 재시도');
        return;
      }

      driver.setData(hmi.uiStatus, response.statusCode);

      if (response.statusCode !== 200) {
        driver.setStringData(hmi.message, 100, '인터넷/VPN 연결 대기중. 10초 후 재시도');
        return;
      }

      let data;
      try {
        data = JSON.parse(body);
      } catch (e) {
        driver.setData(hmi.uiStatus, 0);
        driver.setStringData(hmi.message, 100, '라우터 응답 파싱 실패');
        return;
      }

      if (data.ok === true && data.vpn_ip) {
        driver.setStringData(hmi.router_vpn_ip, 20, data.vpn_ip);
        driver.setStringData(hmi.message, 100, 'VPN 연결 완료');
        return;
      }

      driver.setStringData(hmi.message, 100, 'VPN IP 없음. 10초 후 재시도');
    });
  } catch (err) {
    console.log('Error=>', err.message);
    driver.setData(hmi.uiStatus, 0);
    driver.setStringData(hmi.message, 100, '스크립트 오류');
  }
}
```

`192.168.0.1` 표준 전환 후 URL은 다음으로 바꾼다.

```text
http://192.168.0.1:8080/cgi-bin/router-info
```

---

## 10. LAN IP 변경 및 복구

### 10.1 `192.168.0.1`로 변경

```sh
uci set network.lan.ipaddr='192.168.0.1'
uci set network.lan.netmask='255.255.255.0'
uci commit network
/etc/init.d/network reload
```

변경 후 SSH는 끊긴다. Mac에서 DHCP 갱신:

```bash
sudo ipconfig set en1 DHCP
ipconfig getifaddr en1
```

접속:

```bash
ssh root@192.168.0.1
```

### 10.2 `192.168.1.1`로 복구

```sh
uci set network.lan.ipaddr='192.168.1.1'
uci set network.lan.netmask='255.255.255.0'
uci commit network
/etc/init.d/network reload
```

Mac DHCP 갱신:

```bash
sudo ipconfig set en1 DHCP
ipconfig getifaddr en1
```

접속:

```bash
ssh root@192.168.1.1
```

---

## 11. DHCP 정책

DHCP 변경은 현재 보류했다.

대표님 요구는 장비를 static으로 관리하는 것이다. 따라서 DHCP는 카메라/PLC/HMI 영역과 충돌하지 않게 뒤쪽으로 밀어두는 것이 좋다.

권장 초안:

| IP 범위 | 용도 |
|---|---|
| `192.168.0.1` | 라우터 |
| `192.168.0.10~19` | PLC |
| `192.168.0.20~29` | HMI |
| `192.168.0.30~49` | 기타 고정 장비 |
| `192.168.0.64~99` | IP Camera |
| `192.168.0.150~254` | DHCP 임시 장비 |

OpenWrt DHCP 설정 예:

```sh
uci set dhcp.lan.start='150'
uci set dhcp.lan.limit='105'
uci set dhcp.lan.leasetime='12h'
uci commit dhcp
/etc/init.d/dnsmasq restart
```

주의:

- `192.168.0.65~254`를 DHCP로 쓰면 카메라 `.65` 이상과 충돌할 수 있다.
- 카메라를 `.64~99`로 쓸 계획이면 DHCP는 `.150` 이후가 안전하다.

---

## 12. Hikvision 카메라 IP 이슈

### 12.1 확인된 증상

`192.168.0.1` 실험 중 Mac 상태:

```text
Mac: 192.168.0.194/24
192.168.0.64 ARP incomplete
192.168.0.192 ARP MAC 존재
192.168.0.64 ping 실패
192.168.0.192 ping 실패
```

판단:

- `192.168.0.64`에는 응답 장비가 없었다.
- 카메라는 기존 `192.168.1.64`에 남아 있을 가능성이 높다.
- Hikvision 계열은 기본/초기 IP가 `192.168.1.64`인 경우가 많다.

### 12.2 제조사 앱 없이 확인하는 방법

Mac에 임시 alias 추가:

```bash
sudo ifconfig en1 alias 192.168.1.194 255.255.255.0
```

카메라 확인:

```bash
ping -c 3 192.168.1.64
curl -I --connect-timeout 3 http://192.168.1.64/
open http://192.168.1.64/
```

alias 제거:

```bash
sudo ifconfig en1 -alias 192.168.1.194
```

### 12.3 카메라 목표 설정

```text
IP Address  : 192.168.0.64
Subnet Mask : 255.255.255.0
Gateway     : 192.168.0.1
DNS         : 192.168.0.1 또는 8.8.8.8
DHCP        : Disable
```

대량 세팅 시 동일 기본 IP 카메라를 여러 대 동시에 연결하면 충돌하므로 반드시 1대씩 설정한다.

---

## 13. 제조사 무료 포트패스 구조

현재 무료 포트패스:

```text
http://112.220.220.186:10521/
```

기본 구조:

```text
외부 클라이언트
  -> 112.220.220.186:10521
  -> 제조사 포트패스 서버
  -> 제조사 WireGuard wg
  -> 라우터 wg IP 172.16.1.127
  -> DNAT 10521 -> 192.168.1.1:8080
  -> uhttpd 라우터 웹서버
```

기본 redirect:

```text
firewall.@redirect[0].name='720_web'
firewall.@redirect[0].src='wg'
firewall.@redirect[0].src_dport='10521'
firewall.@redirect[0].dest='lan'
firewall.@redirect[0].dest_ip='192.168.1.1'
firewall.@redirect[0].dest_port='8080'
firewall.@redirect[0].target='DNAT'
```

포트패스 장애 당시 `tcpdump`에서 다음을 확인했다.

```text
172.16.1.254:50925 -> 172.16.1.127:10521 SYN
172.16.1.127:10521 -> 172.16.1.254:50925 RST
```

판단:

- 외부 패킷은 라우터 `wg`까지 도달했다.
- DNAT가 정상 적용되지 않고 라우터 자신이 RST를 보냈다.
- LAN IP 변경/복귀 과정에서 iptables/conntrack 런타임 상태가 꼬였을 가능성이 높다.
- `/etc/init.d/firewall restart` 이후 정상 복구되었다.

복구 명령:

```sh
/etc/init.d/firewall restart
```

복구 확인:

```bash
curl -I --connect-timeout 5 http://112.220.220.186:10521/
```

정상 예:

```text
HTTP/1.1 200 OK
Content-Type: text/html
Content-Length: 522
```

---

## 14. 포트패스를 카메라 Web/RTSP로 전환

무료 포트패스는 하나뿐이므로 동시에 한 대상만 전달한다.

| 모드 | 외부 URL | 내부 대상 |
|---|---|---|
| router | `http://112.220.220.186:10521/` | `192.168.1.1:8080` |
| cam-web | `http://112.220.220.186:10521/` | `192.168.1.64:80` |
| cam-rtsp | `rtsp://112.220.220.186:10521/...` | `192.168.1.64:554` |

### 14.1 전환 스크립트

파일:

```text
/root/switch-portpass.sh
```

내용:

```sh
cat > /root/switch-portpass.sh <<'SH'
#!/bin/sh

MODE="$1"

case "$MODE" in
  router)
    NAME="720_web"
    IP="192.168.1.1"
    PORT="8080"
    ;;
  cam-web)
    NAME="camera_64_web_test"
    IP="192.168.1.64"
    PORT="80"
    ;;
  cam-rtsp)
    NAME="camera_64_rtsp_test"
    IP="192.168.1.64"
    PORT="554"
    ;;
  *)
    echo "Usage: $0 {router|cam-web|cam-rtsp}"
    exit 1
    ;;
esac

echo "[INFO] switch 10521 -> $IP:$PORT ($NAME)"

uci set firewall.@redirect[0].name="$NAME"
uci set firewall.@redirect[0].src='wg'
uci set firewall.@redirect[0].src_dport='10521'
uci set firewall.@redirect[0].dest='lan'
uci set firewall.@redirect[0].dest_ip="$IP"
uci set firewall.@redirect[0].dest_port="$PORT"
uci set firewall.@redirect[0].proto='tcp'
uci set firewall.@redirect[0].target='DNAT'
uci commit firewall
/etc/init.d/firewall restart

uci show firewall.@redirect[0]
iptables -t nat -S | grep 10521
SH

chmod +x /root/switch-portpass.sh
```

사용:

```sh
/root/switch-portpass.sh cam-web
/root/switch-portpass.sh cam-rtsp
/root/switch-portpass.sh router
```

---

## 15. RTSP 터미널 테스트

Python보다 `ffprobe`/`ffmpeg`가 먼저 적합하다.

설치:

```bash
brew install ffmpeg
```

RTSP 연결 확인:

```bash
ffprobe -rtsp_transport tcp -stimeout 5000000 "rtsp://USER:PASS@112.220.220.186:10521/Streaming/Channels/101"
```

스냅샷 저장:

```bash
ffmpeg -rtsp_transport tcp -i "rtsp://USER:PASS@112.220.220.186:10521/Streaming/Channels/101" -frames:v 1 -y snapshot.jpg
```

10초 저장:

```bash
ffmpeg -rtsp_transport tcp -i "rtsp://USER:PASS@112.220.220.186:10521/Streaming/Channels/101" -t 10 -c copy -y test_rtsp.mp4
```

주의:

- 무료 포트패스는 포트 하나만 쓰므로 `-rtsp_transport tcp`를 강제한다.
- RTSP over UDP는 추가 RTP 포트를 요구할 수 있어 실패 가능성이 높다.

---

## 16. 장기 VPN 1:1 매핑 설계

목표:

```text
192.168.0.X <-> 10.<그룹>.<라우터ID>.X
```

예:

```text
192.168.0.10  PLC      <-> 10.1.1.10
192.168.0.20  HMI      <-> 10.1.1.20
192.168.0.64  Camera 1 <-> 10.1.1.64
192.168.0.65  Camera 2 <-> 10.1.1.65
```

GCP WireGuard 서버 peer 예:

```ini
[Peer]
# Router-001
PublicKey = <router-public-key>
AllowedIPs = 10.77.0.2/32, 10.1.1.0/24
```

라우터 NETMAP 지원 확인:

```sh
iptables -t nat -j NETMAP --help >/tmp/netmap_help.txt 2>&1
cat /tmp/netmap_help.txt
```

임시 NETMAP 예:

```sh
iptables-save > /tmp/iptables.before.vpnmap
iptables -t nat -A PREROUTING -i wg2 -d 10.1.1.0/24 -j NETMAP --to 192.168.0.0/24
iptables -t nat -A POSTROUTING -o wg2 -s 192.168.0.0/24 -j NETMAP --to 10.1.1.0/24
```

롤백:

```sh
iptables -t nat -D PREROUTING -i wg2 -d 10.1.1.0/24 -j NETMAP --to 192.168.0.0/24
iptables -t nat -D POSTROUTING -o wg2 -s 192.168.0.0/24 -j NETMAP --to 10.1.1.0/24
```

---

## 17. 완료/미완료 체크리스트

### 17.1 완료

- TC720 SSH 접속 확인
- `wg`, `wg2`, `wg3` 존재 확인
- `wg2` 중앙 VPN 동작 확인
- `wg` 제조사 포트패스 동작 확인
- `uhttpd` 8080 listen 확인
- `/www/cgi-bin/router-info` CGI API 동작 확인
- 라우터 LAN `192.168.0.1` 변경 가능 확인
- 라우터 LAN `192.168.1.1` 복귀 확인
- 제조사 포트패스 `10521 -> 192.168.1.1:8080` 복구 확인
- 포트패스 장애 원인 분석 완료

### 17.2 미완료 / 후속 필요

- 운영용 `router-info`를 `vpn_ip` 단일 키 버전으로 최종 교체
- `wg3` 실패 테스트 API 실제 생성 여부 결정
- HMI 실제 장비에서 200/503/timeout 처리 검증
- Hikvision 카메라 현재 IP 확정
- 카메라 static IP 최종 설정
- 포트패스 cam-web/cam-rtsp 실제 검증
- DHCP 범위 최종 확정
- NETMAP/1:1 VPN 매핑 실제 검증
- 재부팅 후 모든 설정 유지 검증

---

## 18. 빠른 진단 명령 모음

```sh
uci show network.lan
ip addr show br-lan
ip route

curl -I -m 3 http://127.0.0.1:8080/
curl -i -m 3 http://127.0.0.1:8080/cgi-bin/router-info
netstat -ltnp | grep 8080
ps | grep '[u]httpd'

wg show wg
wg show wg2
wg show wg2 latest-handshakes

uci show firewall | grep -A8 -B2 "720_web"
iptables -t nat -S | grep 10521
iptables -t nat -L zone_wg_prerouting -n -v --line-numbers | grep 10521

tcpdump -ni wg tcp port 10521
/etc/init.d/firewall restart
```

---

## 19. 최종 판단

현재까지의 작업은 규격화 가능한 수준까지 진전되었으나, 아직 “양산 최종 설정 완료”는 아니다.

규격화 가능한 항목:

1. `wg2`를 대표 시스템 중앙 WireGuard 인터페이스로 사용한다.
2. PLC/HMI는 `/cgi-bin/router-info`에서 `vpn_ip`를 조회한다.
3. `HTTP 200`일 때만 `vpn_ip`를 신뢰한다.
4. 제조사 `wg` 포트패스는 설정/응급/카메라 RTSP 실험용으로만 사용한다.
5. 장기 운영은 `wg2` 기반 1:1 VPN 매핑으로 이전한다.
6. 카메라/PLC/HMI는 DHCP가 아니라 static IP 기반으로 관리한다.

후속 확정 필요 항목:

1. 카메라 IP 대역 최종 확정
2. DHCP 대역 최종 확정
3. `192.168.0.1` 표준 전환 시점 결정
4. NETMAP 가능 여부 검증
5. HMI 실제 연동 테스트
6. 카메라 RTSP over TCP 포트패스 실험
