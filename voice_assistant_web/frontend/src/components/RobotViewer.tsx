import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { AppLanguage, translations } from '../i18n'
import URDFLoader from '../lib/loaders/URDFLoader.js'
import { truncateLabel } from '../utils/text'

const JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'left_finger']
const DEFAULT_LEFT = [0.0, -0.96, 1.16, 1.57, 0.0, -1.57, 0.02239]
const DEFAULT_RIGHT = [0.0, -0.96, 1.16, 0.0, 0.0, 0.0, 0.02239]
const URDF_URL = 'https://d38qvlnbgujqqd.cloudfront.net/static/urdf/aloha_vx300s.urdf'

type Props = {
  latestAction: number[] | null
  qpos: number[] | null
  mode: string
  currentTask: string | null
  language: AppLanguage
}

export function RobotViewer({ latestAction, qpos, mode, currentTask, language }: Props) {
  const t = translations[language]
  const currentTaskLabel = currentTask ? truncateLabel(currentTask) : t.noActiveTask
  const containerRef = useRef<HTMLDivElement | null>(null)
  const leftRobotRef = useRef<any>(null)
  const rightRobotRef = useRef<any>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xf6f1e8)

    const camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.01, 100)
    camera.position.set(1.25, 1.02, 1.22)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(container.clientWidth, container.clientHeight)
    container.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 0.32, 0)
    controls.enableDamping = true
    controls.minDistance = 0.8
    controls.maxDistance = 3.5

    scene.add(new THREE.AmbientLight(0xffffff, 1.6))
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.8)
    keyLight.position.set(2, 3, 2)
    scene.add(keyLight)
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.8)
    fillLight.position.set(-2, 1, -1)
    scene.add(fillLight)
    scene.add(new THREE.GridHelper(2.5, 18, 0x8d7b68, 0xd8c9b5))

    const loader = new URDFLoader()
    loader.loadMeshCb = (path: string, manager: any, done: (mesh: any) => void) => {
      const stl = new STLLoader(manager)
      stl.load(
        path.trim(),
        (geometry) => {
          geometry.computeVertexNormals()
          done(new THREE.Mesh(geometry, new THREE.MeshNormalMaterial()))
        },
        undefined,
        () => done(new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.04), new THREE.MeshNormalMaterial())),
      )
    }

    const mount = async () => {
      const leftRobot = await new Promise<any>((resolve, reject) => loader.load(URDF_URL, resolve, undefined, reject))
      const rightRobot = await new Promise<any>((resolve, reject) => loader.load(URDF_URL, resolve, undefined, reject))

      leftRobot.position.set(-0.475, 0, 0)
      leftRobot.rotation.x = -Math.PI / 2
      rightRobot.position.set(0.475, 0, 0)
      rightRobot.rotation.y = Math.PI
      rightRobot.rotation.x = Math.PI / 2
      scene.add(leftRobot)
      scene.add(rightRobot)
      leftRobotRef.current = leftRobot
      rightRobotRef.current = rightRobot
      applyArm(DEFAULT_LEFT, DEFAULT_RIGHT)
    }

    const applyArm = (leftValues: number[], rightValues: number[]) => {
      const apply = (robot: any, values: number[]) => {
        if (!robot) return
        JOINT_NAMES.forEach((jointName, index) => {
          if (robot.joints[jointName] && values[index] !== undefined) {
            robot.joints[jointName].setJointValue(values[index])
          }
        })
        robot.updateMatrixWorld(true)
      }
      apply(leftRobotRef.current, leftValues)
      apply(rightRobotRef.current, rightValues)
    }

    mount().catch(console.error)

    let frame = 0
    const render = () => {
      frame = requestAnimationFrame(render)
      controls.update()
      renderer.render(scene, camera)
    }
    render()

    const onResize = () => {
      if (!containerRef.current) return
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    }
    window.addEventListener('resize', onResize)

    ;(container as any).__applyArm = applyArm

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', onResize)
      controls.dispose()
      renderer.dispose()
      leftRobotRef.current = null
      rightRobotRef.current = null
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  useEffect(() => {
    const applyArm = (containerRef.current as any)?.__applyArm as ((left: number[], right: number[]) => void) | undefined
    if (!applyArm) return

    const source = latestAction && latestAction.length >= 14 ? latestAction : qpos && qpos.length >= 14 ? qpos : null
    if (!source) return

    const gripperToFinger = (value: number) => -0.02239 + Math.max(0, Math.min(1, value)) * (0.02239 - -0.02239)
    const left = source.slice(0, 6).concat(gripperToFinger(source[6]))
    const right = source.slice(7, 13).concat(gripperToFinger(source[13]))
    applyArm(left, right)
  }, [latestAction, qpos])

  return (
    <section className="panel robot-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">{t.robotEyebrow}</p>
          <h2>{t.robotTitle}</h2>
        </div>
        <div className="robot-meta">
          <span className="status-pill mode">{mode}</span>
          <span className="robot-task-badge" title={currentTask || t.noActiveTask}>
            {currentTaskLabel}
          </span>
        </div>
      </div>
      <div ref={containerRef} className="robot-canvas" />
    </section>
  )
}
