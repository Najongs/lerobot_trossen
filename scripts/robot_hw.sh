# 로봇 PC별 하드웨어 값 — eval_najy.sh · test_ep1.sh 가 source 한다 (직접 실행하지 않는다).
#
#   TROSSEN_PC=pc1 (기본) | pc2      PC2(새 로봇)에선 ~/.bashrc 에 `export TROSSEN_PC=pc2`
#
# 정한 뒤: ROBOT_LEFT_IP · ROBOT_RIGHT_IP · LEADER_LEFT_IP · LEADER_RIGHT_IP · ROBOT_CAMERAS
# (--robot.cameras 값 그대로) · ARM_IPS(사전 연결 확인용 배열)

TROSSEN_PC=${TROSSEN_PC:-pc1}
case "$TROSSEN_PC" in
  pc1)
    CAM_HIGH=230422273501; CAM_LEFT=230422271234; CAM_RIGHT=230322274369 ;;
  pc2)
    # 10/26 PC2 에서 확인: cam_high 는 그리퍼 없이 바닥 쪽, 왼손목은 손으로 가려 구분
    CAM_HIGH=230422271108; CAM_LEFT=230322270834; CAM_RIGHT=230422271279 ;;
  *) echo "TROSSEN_PC 는 pc1 또는 pc2 (지금 '$TROSSEN_PC')" >&2; exit 2 ;;
esac
# 팔 IP 는 두 PC 가 같다(PC2 는 10/26 아직 유선 미연결 — 다르면 위 case 로 옮긴다)
ROBOT_LEFT_IP=192.168.1.5; ROBOT_RIGHT_IP=192.168.1.4
LEADER_LEFT_IP=192.168.1.3; LEADER_RIGHT_IP=192.168.1.2
ARM_IPS=("$ROBOT_LEFT_IP" "$ROBOT_RIGHT_IP" "$LEADER_LEFT_IP" "$LEADER_RIGHT_IP")

ROBOT_CAMERAS="{
    cam_high: {type: intelrealsense, serial_number_or_name: \"$CAM_HIGH\", width: 640, height: 480, fps: 30},
    cam_left_wrist: {type: intelrealsense, serial_number_or_name: \"$CAM_LEFT\", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: \"$CAM_RIGHT\", width: 640, height: 480, fps: 30}
}"

# 팔 4대 사전 확인 — 유선이 끊긴 채 띄우면 lerobot 이 팔 연결에서 130 s 넘게 기다리다 죽는다(10/15·10/20).
# 1 s 안에 알린다. 로봇에 쓰지 않는다(TCP 연결만 열었다 닫는다). SKIP_NETCHECK=1 로 끈다.
check_arms() {
  [[ "${SKIP_NETCHECK:-0}" == 1 ]] && return 0
  local _down=() _ip
  for _ip in "${ARM_IPS[@]}"; do
    timeout 1 bash -c "exec 3<>/dev/tcp/$_ip/50001" 2>/dev/null || _down+=("$_ip")
  done
  if (( ${#_down[@]} )); then
    echo "!! 팔 컨트롤러 연결 안 됨: ${_down[*]} (TCP 50001) — 유선 링크·스위치·로봇 전원 확인 (ip -br link show enp129s0)" >&2
    exit 4
  fi
}
