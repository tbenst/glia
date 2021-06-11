module ReceptiveField.Main where

import Prelude

import Color (lighten)
import Color.Scheme.MaterialDesign (blueGrey)
import Control.Parallel (parTraverse)
import Data.Array (sortBy, (..))
import Data.Foldable (foldMap)
import Data.Int (toNumber, floor)
import Data.Maybe (fromJust, maybe)
import Data.Newtype (unwrap)
import Data.Set (isEmpty)
import Effect (Effect)
import Effect.Console (log)
import FRP.Behavior (Behavior, animate, fixB, integral', switcher)
import FRP.Behavior.Mouse (buttons)
import FRP.Behavior.Mouse as Mouse
import FRP.Behavior.Time as Time
import FRP.Event.Mouse (Mouse, getMouse, down, move)
import Global (infinity, readFloat)
import Graphics.Canvas (getCanvasElementById, getCanvasHeight, getCanvasWidth, getContext2D, setCanvasHeight, setCanvasWidth)
import Graphics.CanvasAction (loadImageAff, CanvasImageSource)
import Graphics.Drawing (Drawing, circle, fillColor, filled, lineWidth, outlineColor, outlined, rectangle, scale, translate)
import Partial.Unsafe (unsafePartial)
import Effect.Aff (Aff, launchAff_, Fiber)
import Data.Maybe (Maybe(..))
import Effect (Effect)
import Web.DOM.Document (toNonElementParentNode)
import Web.DOM.Element (setAttribute, Element, getAttribute)
import Web.DOM.NonElementParentNode (getElementById)
import Web.HTML (window)
import Web.HTML.HTMLDocument (toDocument)
import Web.HTML.Window (document)

int2img :: String -> Int -> Int -> String
int2img s y x = s <> (show $ y) <> "_" <> (show $ x) <> ".png"

imageStrs :: String -> Int -> Int -> Array String
imageStrs s h w = do
      i <- 0 .. (h-1)
      j <- 0 .. (w-1)
      pure (int2img s i j)

preloadImages :: String -> Int -> Int -> Aff Unit
preloadImages s h w = do
  _ <- parTraverse preLoad (imageStrs s h w)
  pure unit where
    preLoad = \str -> loadImageAff str
    

imgPercentile :: Int -> Int -> Number -> Int
imgPercentile imW x w = floor $ (toNumber imW) * (toNumber x) / w

render :: Element -> String -> Effect Unit
render imgElem path = do
  setAttribute "src" path imgElem