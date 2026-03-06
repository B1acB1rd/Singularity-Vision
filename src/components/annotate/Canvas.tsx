import { useRef, useEffect, useState, type MouseEvent } from 'react';
import type { BoundingBox, ClassInfo, Polygon, Point } from '../../types/project';

export type AnnotationTool = 'move' | 'box' | 'polygon';

interface CanvasProps {
    imageUrl: string;
    boxes: BoundingBox[];
    polygons: Polygon[];
    classes: ClassInfo[];
    currentClassId: number;
    scale: number;
    tool: AnnotationTool;
    onBoxesChange: (boxes: BoundingBox[]) => void;
    onPolygonsChange: (polygons: Polygon[]) => void;
}

export default function Canvas({
    imageUrl,
    boxes,
    polygons,
    classes,
    currentClassId,
    scale,
    tool,
    onBoxesChange,
    onPolygonsChange
}: CanvasProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPos, setStartPos] = useState({ x: 0, y: 0 });
    const [tempBox, setTempBox] = useState<Partial<BoundingBox> | null>(null);
    const [imageObj, setImageObj] = useState<HTMLImageElement | null>(null);

    // Polygon drawing state
    const [currentPolygonPoints, setCurrentPolygonPoints] = useState<Point[]>([]);
    const [hoverPoint, setHoverPoint] = useState<Point | null>(null);

    // Reset polygon drawing state when image changes
    useEffect(() => {
        setCurrentPolygonPoints([]);
        setHoverPoint(null);
    }, [imageUrl]);

    // Load image
    useEffect(() => {
        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            setImageObj(img);
        };
    }, [imageUrl]);

    // Draw canvas
    useEffect(() => {
        if (!canvasRef.current || !imageObj || !containerRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Set canvas size to match scaled image
        const scaledWidth = imageObj.width * scale;
        const scaledHeight = imageObj.height * scale;

        canvas.width = scaledWidth;
        canvas.height = scaledHeight;

        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Image
        ctx.drawImage(imageObj, 0, 0, scaledWidth, scaledHeight);

        // Draw Boxes
        boxes.forEach(box => {
            const x = box.x * scaledWidth;
            const y = box.y * scaledHeight;
            const w = box.width * scaledWidth;
            const h = box.height * scaledHeight;

            const cls = classes.find(c => c.id === box.classId);
            const color = cls ? cls.color : '#fff';

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            // Draw Label
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.7;
            const text = cls ? cls.name : `Class ${box.classId}`;
            const textWidth = ctx.measureText(text).width + 6;
            ctx.fillRect(x, y - 20, textWidth, 20);
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = '#fff';
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(text, x + 3, y - 6);
        });

        // Draw Polygons
        polygons.forEach(poly => {
            if (poly.points.length < 2) return;

            const cls = classes.find(c => c.id === poly.classId);
            const color = cls ? cls.color : '#fff';

            ctx.beginPath();
            const firstPoint = poly.points[0];
            ctx.moveTo(firstPoint.x * scaledWidth, firstPoint.y * scaledHeight);

            for (let i = 1; i < poly.points.length; i++) {
                const pt = poly.points[i];
                ctx.lineTo(pt.x * scaledWidth, pt.y * scaledHeight);
            }

            if (poly.closed) {
                ctx.closePath();
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.3;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw vertices
            poly.points.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x * scaledWidth, pt.y * scaledHeight, 4, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1;
                ctx.stroke();
            });

            // Draw label at first point
            if (poly.points.length > 0) {
                const labelPt = poly.points[0];
                const text = cls ? cls.name : `Class ${poly.classId}`;
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.8;
                const textWidth = ctx.measureText(text).width + 6;
                ctx.fillRect(labelPt.x * scaledWidth - 3, labelPt.y * scaledHeight - 24, textWidth, 20);
                ctx.globalAlpha = 1.0;
                ctx.fillStyle = '#fff';
                ctx.font = '12px Inter, sans-serif';
                ctx.fillText(text, labelPt.x * scaledWidth, labelPt.y * scaledHeight - 8);
            }
        });

        // Draw current polygon being drawn
        if (currentPolygonPoints.length > 0) {
            const cls = classes.find(c => c.id === currentClassId);
            const color = cls ? cls.color : '#22c55e';

            ctx.beginPath();
            const firstPoint = currentPolygonPoints[0];
            ctx.moveTo(firstPoint.x * scaledWidth, firstPoint.y * scaledHeight);

            for (let i = 1; i < currentPolygonPoints.length; i++) {
                const pt = currentPolygonPoints[i];
                ctx.lineTo(pt.x * scaledWidth, pt.y * scaledHeight);
            }

            // Draw line to hover point
            if (hoverPoint) {
                ctx.lineTo(hoverPoint.x * scaledWidth, hoverPoint.y * scaledHeight);
            }

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw vertices
            currentPolygonPoints.forEach((pt, idx) => {
                ctx.beginPath();
                ctx.arc(pt.x * scaledWidth, pt.y * scaledHeight, idx === 0 ? 8 : 5, 0, Math.PI * 2);
                ctx.fillStyle = idx === 0 ? '#22c55e' : color;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        }

        // Draw Temp Box
        if (tempBox && isDrawing && tool === 'box') {
            const x = (tempBox.x || 0) * scaledWidth;
            const y = (tempBox.y || 0) * scaledHeight;
            const w = (tempBox.width || 0) * scaledWidth;
            const h = (tempBox.height || 0) * scaledHeight;

            const cls = classes.find(c => c.id === currentClassId);
            const color = cls ? cls.color : '#fff';

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
        }

    }, [imageObj, boxes, polygons, tempBox, scale, classes, currentClassId, isDrawing, currentPolygonPoints, hoverPoint, tool]);

    const handleMouseDown = (e: MouseEvent) => {
        if (!canvasRef.current || !imageObj) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = (e.clientX - rect.left) / scale;
        const y = (e.clientY - rect.top) / scale;
        const normalizedX = x / imageObj.width;
        const normalizedY = y / imageObj.height;

        if (tool === 'box') {
            setStartPos({ x, y });
            setIsDrawing(true);
            setTempBox({
                x: normalizedX,
                y: normalizedY,
                width: 0,
                height: 0,
                classId: currentClassId
            });
        } else if (tool === 'polygon') {
            // Check if clicking near first point to close polygon
            if (currentPolygonPoints.length >= 3) {
                const firstPt = currentPolygonPoints[0];
                const distToFirst = Math.sqrt(
                    Math.pow((normalizedX - firstPt.x) * imageObj.width, 2) +
                    Math.pow((normalizedY - firstPt.y) * imageObj.height, 2)
                );

                if (distToFirst < 15) {
                    // Close polygon
                    const newPolygon: Polygon = {
                        id: crypto.randomUUID(),
                        points: currentPolygonPoints,
                        classId: currentClassId,
                        className: classes.find(c => c.id === currentClassId)?.name || 'Unknown',
                        closed: true
                    };
                    onPolygonsChange([...polygons, newPolygon]);
                    setCurrentPolygonPoints([]);
                    return;
                }
            }

            // Add new point
            setCurrentPolygonPoints(prev => [...prev, { x: normalizedX, y: normalizedY }]);
        }
    };

    const handleMouseMove = (e: MouseEvent) => {
        if (!canvasRef.current || !imageObj) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const currentX = (e.clientX - rect.left) / scale;
        const currentY = (e.clientY - rect.top) / scale;

        if (tool === 'polygon' && currentPolygonPoints.length > 0) {
            setHoverPoint({
                x: currentX / imageObj.width,
                y: currentY / imageObj.height
            });
        }

        if (!isDrawing || tool !== 'box') return;

        const x = Math.min(startPos.x, currentX);
        const y = Math.min(startPos.y, currentY);
        const w = Math.abs(currentX - startPos.x);
        const h = Math.abs(currentY - startPos.y);

        setTempBox({
            x: x / imageObj.width,
            y: y / imageObj.height,
            width: w / imageObj.width,
            height: h / imageObj.height,
            classId: currentClassId
        });
    };

    const handleMouseUp = () => {
        if (!isDrawing || !tempBox || !imageObj || tool !== 'box') return;

        if ((tempBox.width || 0) > 0.01 && (tempBox.height || 0) > 0.01) {
            const newBox: BoundingBox = {
                id: crypto.randomUUID(),
                x: tempBox.x || 0,
                y: tempBox.y || 0,
                width: tempBox.width || 0,
                height: tempBox.height || 0,
                classId: currentClassId,
                className: classes.find(c => c.id === currentClassId)?.name || 'Unknown'
            };
            onBoxesChange([...boxes, newBox]);
        }

        setIsDrawing(false);
        setTempBox(null);
    };

    const handleDoubleClick = () => {
        // Complete polygon on double-click (without closing)
        if (tool === 'polygon' && currentPolygonPoints.length >= 2) {
            const newPolygon: Polygon = {
                id: crypto.randomUUID(),
                points: currentPolygonPoints,
                classId: currentClassId,
                className: classes.find(c => c.id === currentClassId)?.name || 'Unknown',
                closed: currentPolygonPoints.length >= 3
            };
            onPolygonsChange([...polygons, newPolygon]);
            setCurrentPolygonPoints([]);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Escape' && currentPolygonPoints.length > 0) {
            setCurrentPolygonPoints([]);
            setHoverPoint(null);
        }
    };

    return (
        <div
            ref={containerRef}
            className="canvas-container inline-block outline-none"
            tabIndex={0}
            onKeyDown={handleKeyDown}
        >
            {imageObj ? (
                <canvas
                    ref={canvasRef}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={() => {
                        handleMouseUp();
                        setHoverPoint(null);
                    }}
                    onDoubleClick={handleDoubleClick}
                    style={{
                        cursor: tool === 'box' ? 'crosshair' :
                            tool === 'polygon' ? 'crosshair' : 'default'
                    }}
                />
            ) : (
                <div className="flex items-center justify-center p-12 text-gray-500">
                    Loading Image...
                </div>
            )}
        </div>
    );
}
