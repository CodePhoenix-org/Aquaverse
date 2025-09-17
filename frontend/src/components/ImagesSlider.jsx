import { cn } from '../lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';

export const ImagesSlider = ({
  images,
  children,
  overlay = true,
  overlayClassName,
  className,
  autoplay = true,
  direction = 'up',
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadedImages, setLoadedImages] = useState([]);

  const handleNext = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex + 1 === images.length ? 0 : prevIndex + 1
    );
  };

  const handlePrevious = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex - 1 < 0 ? images.length - 1 : prevIndex - 1
    );
  };

  useEffect(() => {
    const loadImages = () => {
      setLoading(true);
      const loadPromises = images.map((image) => {
        return new Promise((resolve, reject) => {
          const img = new Image();
          img.src = image;
          img.onload = () => {
            console.log('Image loaded:', image);
            resolve(image);
          };
          img.onerror = (error) => {
            console.error('Failed to load image:', image, error);
            reject(error);
          };
        });
      });

      Promise.all(loadPromises)
        .then((loadedImages) => {
          console.log('All images loaded:', loadedImages);
          setLoadedImages(loadedImages);
          setLoading(false);
        })
        .catch((error) => console.error('Failed to load images', error));
    };
    loadImages();
  }, [images]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'ArrowRight') {
        handleNext();
      } else if (event.key === 'ArrowLeft') {
        handlePrevious();
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    let interval;
    if (autoplay) {
      interval = setInterval(() => {
        handleNext();
      }, 5000);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      clearInterval(interval);
    };
  }, [autoplay]);

  const slideVariants = {
    initial: {
      scale: 0,
      opacity: 0,
      rotateX: 45,
    },
    visible: {
      scale: 1,
      rotateX: 0,
      opacity: 1,
      transition: {
        duration: 0.5,
        ease: [0.645, 0.045, 0.355, 1.0],
      },
    },
    upExit: {
      opacity: 1,
      y: '-150%',
      transition: {
        duration: 1,
      },
    },
    downExit: {
      opacity: 1,
      y: '150%',
      transition: {
        duration: 1,
      },
    },
  };

  const areImagesLoaded = loadedImages.length > 0;

  return (
    <div
      className={cn(
        'overflow-hidden h-full w-full relative bg-gradient-to-br from-blue-900 via-cyan-900 to-blue-800',
        className
      )}
      style={{
        perspective: '1000px',
      }}
    >
      {/* Fallback background if no images loaded */}
      {!areImagesLoaded && (
        <div className="h-full w-full bg-gradient-to-br from-blue-900 via-cyan-900 to-blue-800 flex items-center justify-center">
          <div className="text-white text-center">
            <div className="w-16 h-16 border-4 border-white/30 border-t-white rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-lg">Loading ocean visuals...</p>
          </div>
        </div>
      )}
      
      {areImagesLoaded && overlay && (
        <div className={cn('absolute inset-0 z-40', overlayClassName)} />
      )}
      {areImagesLoaded && (
        <AnimatePresence>
          <motion.img
            key={currentIndex}
            src={loadedImages[currentIndex]}
            initial="initial"
            animate="visible"
            exit={direction === 'up' ? 'upExit' : 'downExit'}
            variants={slideVariants}
            className="image h-full w-full absolute inset-0 object-cover object-center"
            alt={`Ocean image ${currentIndex + 1}`}
          />
        </AnimatePresence>
      )}
      {areImagesLoaded && children}
    </div>
  );
};