import cv2
import numpy as np
from pathlib import Path
import os
import sys

class SquareImageCropper:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        self.display_image = self.image.copy()
        self.original_image = self.image.copy()
        
        # Variables para el recorte cuadrado
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.square_size = 0
        
        # Configuración de ventana
        self.window_name = "Recortador Cuadrado - Click y arrastra | ESC: salir | ENTER: guardar | R: reset"
        
    def draw_square(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
    
    def calculate_square(self):
        """Calcula las coordenadas de un cuadrado perfecto"""
        if self.start_point is None or self.end_point is None:
            return None
        
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Calcular el tamaño del cuadrado basado en la distancia más grande
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        size = max(dx, dy)
        
        # Determinar la dirección del arrastre
        dir_x = 1 if x2 >= x1 else -1
        dir_y = 1 if y2 >= y1 else -1
        
        # Calcular las coordenadas del cuadrado
        end_x = x1 + (size * dir_x)
        end_y = y1 + (size * dir_y)
        
        # Asegurar que el cuadrado está dentro de los límites de la imagen
        h, w = self.image.shape[:2]
        
        if end_x < 0:
            x1 = size
            end_x = 0
        elif end_x > w:
            end_x = w
            x1 = w - size
            
        if end_y < 0:
            y1 = size
            end_y = 0
        elif end_y > h:
            end_y = h
            y1 = h - size
        
        # Ajustar si el cuadrado es demasiado grande
        if x1 < 0:
            x1 = 0
            end_x = min(size, w)
        if y1 < 0:
            y1 = 0
            end_y = min(size, h)
        
        return (min(x1, end_x), min(y1, end_y), max(x1, end_x), max(y1, end_y))
    
    def update_display(self):
        """Actualiza la visualización con el cuadrado dibujado"""
        self.display_image = self.original_image.copy()
        
        square_coords = self.calculate_square()
        if square_coords:
            x1, y1, x2, y2 = square_coords
            
            # Dibujar el cuadrado
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar las esquinas
            corner_size = 10
            cv2.line(self.display_image, (x1, y1), (x1 + corner_size, y1), (0, 255, 0), 3)
            cv2.line(self.display_image, (x1, y1), (x1, y1 + corner_size), (0, 255, 0), 3)
            cv2.line(self.display_image, (x2, y1), (x2 - corner_size, y1), (0, 255, 0), 3)
            cv2.line(self.display_image, (x2, y1), (x2, y1 + corner_size), (0, 255, 0), 3)
            cv2.line(self.display_image, (x1, y2), (x1 + corner_size, y2), (0, 255, 0), 3)
            cv2.line(self.display_image, (x1, y2), (x1, y2 - corner_size), (0, 255, 0), 3)
            cv2.line(self.display_image, (x2, y2), (x2 - corner_size, y2), (0, 255, 0), 3)
            cv2.line(self.display_image, (x2, y2), (x2, y2 - corner_size), (0, 255, 0), 3)
            
            # Mostrar dimensiones
            size = x2 - x1
            text = f"{size}x{size} px"
            cv2.putText(self.display_image, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Oscurecer el área fuera del cuadrado
            mask = np.zeros_like(self.display_image)
            mask[y1:y2, x1:x2] = 255
            overlay = self.display_image.copy()
            overlay[mask == 0] = overlay[mask == 0] * 0.5
            self.display_image = overlay.astype(np.uint8)
    
    def crop_and_save(self, output_path=None):
        """Recorta y guarda la imagen"""
        square_coords = self.calculate_square()
        if square_coords is None:
            print("⚠️  No se ha seleccionado ningún área")
            return False
        
        x1, y1, x2, y2 = square_coords
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            print("⚠️  El área seleccionada es demasiado pequeña")
            return False
        
        cropped = self.original_image[y1:y2, x1:x2]
        
        if output_path is None:
            # Generar nombre de archivo automático
            path = Path(self.image_path)
            output_path = path.parent / f"{path.stem}_cropped{path.suffix}"
        
        cv2.imwrite(str(output_path), cropped)
        print(f"✅ Imagen guardada: {output_path}")
        print(f"   Dimensiones: {x2-x1}x{y2-y1} px")
        return True
    
    def reset(self):
        """Reinicia la selección"""
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.display_image = self.original_image.copy()
    
    def run(self):
        """Ejecuta el recortador interactivo"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.draw_square)
        
        print("\n" + "="*60)
        print("🖼️  RECORTADOR CUADRADO DE IMÁGENES")
        print("="*60)
        print("\n📋 Instrucciones:")
        print("   • Click y arrastra para seleccionar el área cuadrada")
        print("   • ENTER: Guardar imagen recortada")
        print("   • R: Reiniciar selección")
        print("   • ESC: Salir sin guardar")
        print("="*60 + "\n")
        
        while True:
            self.update_display()
            cv2.imshow(self.window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            # ESC - Salir
            if key == 27:
                print("❌ Operación cancelada")
                break
            
            # ENTER - Guardar
            elif key == 13:
                if self.crop_and_save():
                    break
            
            # R - Reset
            elif key == ord('r') or key == ord('R'):
                self.reset()
                print("🔄 Selección reiniciada")
        
        cv2.destroyAllWindows()


class BatchSquareCropper:
    """Recorta múltiples imágenes de una carpeta"""
    def __init__(self, input_folder, output_folder=None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder) if output_folder else self.input_folder / "cropped"
        self.output_folder.mkdir(exist_ok=True)
        
        # Extensiones soportadas
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def get_image_files(self):
        """Obtiene lista de archivos de imagen"""
        images = []
        for ext in self.supported_extensions:
            images.extend(self.input_folder.glob(f"*{ext}"))
            images.extend(self.input_folder.glob(f"*{ext.upper()}"))
        return sorted(images)
    
    def process_batch(self):
        """Procesa todas las imágenes de la carpeta"""
        images = self.get_image_files()
        
        if not images:
            print(f"⚠️  No se encontraron imágenes en: {self.input_folder}")
            return
        
        print(f"\n📁 Encontradas {len(images)} imágenes")
        print(f"💾 Las imágenes se guardarán en: {self.output_folder}\n")
        
        for idx, image_path in enumerate(images, 1):
            print(f"\n[{idx}/{len(images)}] Procesando: {image_path.name}")
            
            try:
                cropper = SquareImageCropper(str(image_path))
                output_path = self.output_folder / image_path.name
                
                cropper.run()
                
            except Exception as e:
                print(f"❌ Error al procesar {image_path.name}: {e}")
                continue
        
        print("\n" + "="*60)
        print(f"✅ Proceso completado!")
        print(f"📁 Imágenes guardadas en: {self.output_folder}")
        print("="*60)


# ==================== EJEMPLOS DE USO ====================
if __name__ == '__main__':
    # OPCIÓN 1: Recortar una sola imagen
    cropper = SquareImageCropper(sys.argv[1])
    cropper.run()
    
    # OPCIÓN 2: Recortar especificando el archivo de salida
    # cropper = SquareImageCropper('imagen.jpg')
    # cropper.run()
    # cropper.crop_and_save('recorte_final.jpg')
    
    # OPCIÓN 3: Procesar múltiples imágenes de una carpeta
    # batch = BatchSquareCropper(
    #     input_folder='imagenes_originales',
    #     output_folder='imagenes_recortadas'
    # )
    # batch.process_batch()
