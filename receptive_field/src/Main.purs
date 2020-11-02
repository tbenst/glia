module ReceptiveField.Main where

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

int2img :: String -> Int -> Int -> String
int2img s x y = s <> (show $ y) <> "_" <> (show $ x) <> ".png"

-- 
imgPercentile :: Int -> Int -> Number -> Int
imgPercentile imW x w = floor $ (toNumber imW) * (toNumber x) / w

render :: Element -> String -> Effect Unit
render imgElem path = do
  setAttribute "src" path imgElem