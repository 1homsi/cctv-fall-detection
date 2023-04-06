import { useEffect, useState } from "react"

export function useCursor() {
  const [hovered, setHover] = useState(false)
  useEffect(() => void (document.body.style.cursor = hovered ? "pointer" : "auto"), [hovered])

  return {
    onPointerOver: () => setHover(true),
    onPointerOut: () => setHover(false)
  }
}
