module ReceptiveField.Baceup where

import Prelude

import Color (lighten)
import Color.Scheme.MaterialDesign (blueGrey)
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
import Graphics.Drawing (Drawing, circle, fillColor, filled, lineWidth, outlineColor, outlined, rectangle, scale, translate)
import Partial.Unsafe (unsafePartial)

import Data.Maybe (Maybe(..))
import Effect (Effect)
import Web.DOM.Document (toNonElementParentNode)
import Web.DOM.Element (setAttribute, Element, getAttribute)
import Web.DOM.NonElementParentNode (getElementById)
import Web.HTML (window)
import Web.HTML.HTMLDocument (toDocument)
import Web.HTML.Window (document)

-- https://stackoverflow.com/questions/62001511/getelementbyid-in-purescript
type Circle = { x :: Number, y :: Number, size :: Number }

int2img :: Int -> Int -> String
int2img x y = "images/retina2pixel/" <> (show $ 2*y) <> "_" <> (show $ 2*x) <> ".png"

-- 
imgPercentile :: Int -> Int -> Number -> Int
imgPercentile imW x w = floor $ (toNumber imW) * (toNumber x) / w

scene :: Mouse -> { w :: Number, h :: Number } -> Behavior String
scene mouse { w, h } = maybe "/retina2pixel/16_16.png"
  (\{x, y} -> int2img (widthPerc x w) (heightPerc y h))
  <$> (Mouse.position mouse) where
    heightPerc = imgPercentile 32
    widthPerc = imgPercentile 32

render :: Element -> String -> Effect Unit
render imgElem path = do
  setAttribute "src" path imgElem


main :: Effect Unit
main = do
  log "Hello"
  let w = 200.5
  let h = 200.5
  mouse <- getMouse
  win ← window
  doc ← document win
  maybeElement ← getElementById "retina2pixel" $ toNonElementParentNode $ toDocument  doc
  case maybeElement of
    Nothing → log "couldn't find img by ID"
    Just elem → do
      mH <- getAttribute "height" elem
      let h = readFloat (unsafePartial fromJust (mH))
      mW <- getAttribute "width" elem
      let w  = readFloat (unsafePartial (fromJust mW))
      -- _ <- setAttribute "height" (show h) elem
      -- _ <- setAttribute "width" (show w) elem
      _ <- animate (scene mouse {w, h}) (render elem)
      pure unit
  pure unit
